"""K線氣象台：用 Kronos-small 預報 stocks.json 裡每檔台股未來 20 個交易日，輸出 docs/forecast.json。

每檔做兩件事：
1. 用最近 120 個交易日的日 K 生成 32 條未來 20 日走勢，算上漲機率與 10/50/90 分位。
2. 滾動回測：站在 20、40 … 120 個交易日前做同樣的預報，跟實際收盤比，算方向命中與落在區間的比例。

股價來源：FinMind（台股日成交資訊，免 token 可用，有 FINMIND_TOKEN 額度較高），失敗時改用 Yahoo 財經。
回看期用 120 日：實測 400 日時 Kronos 對大漲過的股票會嚴重偏空（景碩預測 -70%，實際 -22%）。
"""
import json, os, sys, time, datetime as dt, urllib.request, urllib.parse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ.get("KRONOS_DIR", ROOT / "Kronos"))))
from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference  # noqa: E402

LOOKBACK, PRED, PATHS = 120, 20, 32
CUTS = (20, 40, 60, 80, 100, 120)
TZ = dt.timezone(dt.timedelta(hours=8))
STOCKS = json.loads((ROOT / "stocks.json").read_text(encoding="utf-8"))
torch.manual_seed(42)
torch.set_num_threads(os.cpu_count() or 2)


def http_json(url, headers=None):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kline-weather/1.0", **(headers or {})})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 + attempt * 3)
    raise err


def finmind(code):
    start = (dt.date.today() - dt.timedelta(days=500)).isoformat()
    q = urllib.parse.urlencode({"dataset": "TaiwanStockPrice", "data_id": code, "start_date": start})
    tok = os.environ.get("FINMIND_TOKEN", "").strip()
    d = http_json("https://api.finmindtrade.com/api/v4/data?" + q, {"Authorization": f"Bearer {tok}"} if tok else None)
    rows = [(dt.date.fromisoformat(r["date"]), r["open"], r["max"], r["min"], r["close"], r["Trading_Volume"]) for r in d.get("data", [])]
    return [r for r in rows if r[5] > 0 and r[1] > 0]


def yahoo(code):
    for suf in (".TW", ".TWO"):
        try:
            r = http_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{code}{suf}?range=2y&interval=1d")["chart"]["result"][0]
        except Exception:  # noqa: BLE001
            continue
        q = r["indicators"]["quote"][0]
        rows = [(dt.datetime.fromtimestamp(t, TZ).date(), q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]) for i, t in enumerate(r["timestamp"])]
        rows = [x for x in rows if None not in x and x[5] > 0]
        if rows:
            return rows
    return []


def market_of(code):
    try:
        d = http_json("https://api.finmindtrade.com/api/v4/data?" + urllib.parse.urlencode({"dataset": "TaiwanStockInfo", "data_id": code}))
        t = {x.get("type") for x in d.get("data", [])}
        return "上櫃" if "tpex" in t else "上市"
    except Exception:  # noqa: BLE001
        return ""


def load(code):
    need = LOOKBACK + max(CUTS) + 5
    for src, fn in (("FinMind", finmind), ("Yahoo", yahoo)):
        try:
            rows = fn(code)
        except Exception as e:  # noqa: BLE001
            print(f"  {src} failed: {e}", flush=True)
            continue
        if len(rows) >= need:
            return src, rows
        print(f"  {src} only {len(rows)} rows", flush=True)
    raise RuntimeError(f"no data for {code}")


def stamps(days):
    return np.array([[0, 0, d.weekday(), d.day, d.month] for d in days], np.float32)


def future_days(last, n):
    out, d = [], last
    while len(out) < n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def forecast(tok, model, rows):
    """rows: [(date, o, h, l, c, v)] → (future dates, closes PATHS x PRED)，與官方 KronosPredictor 相同的正規化。"""
    days = [r[0] for r in rows]
    x = np.array([[o, h, l, c, v, v * (o + h + l + c) / 4] for _, o, h, l, c, v in rows], np.float32)
    mean, std = x.mean(0), x.std(0)
    xn = np.clip((x - mean) / (std + 1e-5), -5, 5)
    fut = future_days(days[-1], PRED)
    rep = lambda a: torch.from_numpy(np.repeat(a[None], PATHS, 0))  # noqa: E731
    p = auto_regressive_inference(tok, model, rep(xn), rep(stamps(days)), rep(stamps(fut)), 512, PRED, 5, 1.0, 0, 0.9, 1)[:, -PRED:, :]
    p = p * (std + 1e-5) + mean
    return fut, p[:, :, 3]


r2 = lambda a: [round(float(v), 2) for v in a]  # noqa: E731


def main():
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()
    out = {"generated": dt.datetime.now(TZ).strftime("%Y-%m-%d %H:%M"), "model": "Kronos-small",
           "lookback": LOOKBACK, "pred": PRED, "paths": PATHS, "stocks": []}
    for code, name in STOCKS:
        t0 = time.time()
        src, rows = load(code)
        hist = rows[-LOOKBACK:]
        fut, c = forecast(tok, model, hist)
        last = hist[-1][4]
        tests = []
        for cut in CUTS:
            bh = rows[-LOOKBACK - cut:-cut]
            _, bc = forecast(tok, model, bh)
            base = bh[-1][4]
            act = np.array([r[4] for r in rows[-cut:][:PRED]])
            n = len(act)
            med, lo, hi = np.median(bc, 0)[:n], np.percentile(bc, 10, 0)[:n], np.percentile(bc, 90, 0)[:n]
            t = {"start": bh[-1][0].isoformat(), "base": round(base, 2),
                 "predRet": round(float(med[-1] / base - 1), 4), "realRet": round(float(act[-1] / base - 1), 4),
                 "hit": int((med[-1] > base) == (act[-1] > base)), "inBand": round(float(((act >= lo) & (act <= hi)).mean()), 2)}
            if cut == CUTS[0]:
                t.update({"days": [r[0].isoformat() for r in rows[-cut:][:PRED]], "actual": r2(act), "median": r2(med), "p10": r2(lo), "p90": r2(hi)})
            tests.append(t)
        s = {"code": code, "name": name, "market": market_of(code), "source": src, "last": round(last, 2), "lastDate": hist[-1][0].isoformat(),
             "history": [[d.isoformat(), round(o, 2), round(h, 2), round(l, 2), round(cl, 2), int(v)] for d, o, h, l, cl, v in hist],
             "future": [d.isoformat() for d in fut], "median": r2(np.median(c, 0)), "p10": r2(np.percentile(c, 10, 0)), "p90": r2(np.percentile(c, 90, 0)),
             "paths": [r2(c[i]) for i in range(10)], "upProb": round(float((c[:, -1] > last).mean()), 3),
             "expRet": round(float(np.median(c[:, -1]) / last - 1), 4), "tests": tests}
        out["stocks"].append(s)
        print(f"{code} {name} [{src}] 上漲機率 {s['upProb']:.0%} 中位數 {s['expRet']:+.1%} 命中 {sum(t['hit'] for t in tests)}/{len(CUTS)} {time.time() - t0:.0f}s", flush=True)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "forecast.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("wrote docs/forecast.json")


if __name__ == "__main__":
    main()
