import os
import json
import math
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from datetime import datetime, timedelta

HISTORY_FILE = "history.csv"
CHART_FILE = "history.png"
TZ_BJ = pytz.timezone('Asia/Shanghai')

def color_text(val, text):
    if val > 0:
        return f'<font color="#d93025">{text}</font>'  # 涨-红
    elif val < 0:
        return f'<font color="#188038">{text}</font>'  # 跌-绿
    return text

def format_diff(diff_pct):
    sign = "+" if diff_pct > 0 else ""
    return color_text(diff_pct, f"{sign}{diff_pct:.2f}%")

def format_pnl_str(pnl_val, pct_val):
    sign = "+" if pnl_val > 0 else ""
    return color_text(pnl_val, f"{sign}¥{pnl_val:,.0f} ({sign}{pct_val:.2f}%)")

def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_market_history(assets):
    tickers = [item["ticker"] for item in assets.values()]
    tickers += ["USDCNY=X", "USDT-USD", "USDC-USD"]
    df = yf.download(tickers=tickers, period="1mo", interval="1d", progress=False)['Close']
    df = df.ffill().bfill()
    latest = df.iloc[-1]
    return latest

def load_or_init_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["timestamp", "net_assets", "principal", "total_assets"])

def find_closest_record(df_hist, target_dt):
    """在历史记录中，寻找与 target_dt 时间差最小的那一条记录"""
    if df_hist.empty:
        return None
    time_diffs = (df_hist["dt"] - target_dt).abs()
    best_idx = time_diffs.idxmin()
    return df_hist.loc[best_idx]
def get_baseline_pnl(df_hist, now_net, now_dt):
    """
    精准锚定 0 点寻找最近历史快照计算 日/周/月/年 盈亏
    """
    if df_hist.empty or len(df_hist) < 1:
        return (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

    df_hist["dt"] = pd.to_datetime(df_hist["timestamp"]).dt.tz_convert('Asia/Shanghai')

    report_dt = now_dt
    
    # 2. 构造四个锚点时间（北京时间）
    target_day = report_dt.replace(hour=0, minute=0, second=0, microsecond=0)      # 今日 00:00
    target_week = target_day - timedelta(days=report_dt.weekday())                 # 本周一 00:00
    target_month = target_day.replace(day=1)                                       # 本月 1 日 00:00
    target_year = target_day.replace(month=1, day=1)                               # 本年 1 月 1 日 00:00

    # 3. 寻找最接近锚点的记录
    rec_day = find_closest_record(df_hist, target_day)
    rec_week = find_closest_record(df_hist, target_week)
    rec_month = find_closest_record(df_hist, target_month)
    rec_year = find_closest_record(df_hist, target_year)

    def calc_diff(record):
        if record is None:
            return 0.0, 0.0
        base_val = float(record["net_assets"])
        diff = now_net - base_val
        pct = (diff / base_val) * 100 if base_val > 0 else 0.0
        return diff, pct

    return calc_diff(rec_day), calc_diff(rec_week), calc_diff(rec_month), calc_diff(rec_year)

def plot_performance_chart(df_hist):
    if len(df_hist) < 2:
        return False

    # 转换时区
    df_hist["dt"] = pd.to_datetime(df_hist["timestamp"]).dt.tz_convert('Asia/Shanghai')
    
    # 设置样式
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    # 绘制折线
    ax.plot(df_hist["dt"], df_hist["net_assets"], label="Net Assets", color="#1a73e8", linewidth=2.5)
    ax.plot(df_hist["dt"], df_hist["principal"], label="Baseline", color="#80868b", linewidth=1.8, linestyle="--")

    # 填充颜色区域代表超额盈亏
    ax.fill_between(df_hist["dt"], df_hist["net_assets"], df_hist["principal"], 
                    where=(df_hist["net_assets"] >= df_hist["principal"]),
                    facecolor='#ea4335', alpha=0.15, interpolate=True)
    ax.fill_between(df_hist["dt"], df_hist["net_assets"], df_hist["principal"], 
                    where=(df_hist["net_assets"] < df_hist["principal"]),
                    facecolor='#34a853', alpha=0.15, interpolate=True)

    # 日期轴格式化
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d', tz=TZ_BJ))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'¥{x:,.0f}'))

    ax.set_title("Portfolio Growth vs. Cash Baseline", fontsize=14, pad=12, weight='bold')
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
    plt.tight_layout()
    plt.savefig(CHART_FILE)
    plt.close()
    return True

def evaluate():
    cfg = load_config()
    assets = cfg["assets"]
    latest = fetch_market_history(assets)

    usd_cny_now = float(latest["USDCNY=X"])
    usdt_usd_now = float(latest["USDT-USD"])
    usdc_usd_now = float(latest["USDC-USD"])

    total_assets_now = 0.0
    asset_states = {}

    for code, info in assets.items():
        ticker = info["ticker"]
        qty = info["qty"]
        curr_type = info["currency"]
        p_now = float(latest[ticker])

        if curr_type == "CNY":
            local_p_now = p_now
            cny_val_now = qty * p_now
        elif curr_type == "USDT":
            local_p_now = p_now / usdt_usd_now
            cny_val_now = qty * p_now * usd_cny_now
        elif curr_type == "USDC":
            local_p_now = p_now / usdc_usd_now
            cny_val_now = qty * p_now * usd_cny_now

        total_assets_now += cny_val_now
        asset_states[code] = {
            "name": info["name"],
            "currency": curr_type,
            "group": info["group"],
            "target_w": info["target_w"],
            "qty": qty,
            "local_price": local_p_now,
            "local_val": qty * local_p_now,
            "cny_val": cny_val_now,
        }

    debt = cfg["debt_cny"]
    net_now = total_assets_now - debt
    init_cap = cfg["initial_capital_cny"]
    leverage = total_assets_now / net_now if net_now > 0 else 0

    # 历史记录比对
    now_dt = datetime.now(TZ_BJ)
    df_hist = load_or_init_history()
    # 接收新增的 year_pnl
    day_pnl, week_pnl, month_pnl, year_pnl = get_baseline_pnl(df_hist, net_now, now_dt)

    total_pnl = net_now - init_cap
    total_pnl_pct = (total_pnl / init_cap) * 100

    # 保存最新快照（剔除临时计算列 dt）
    new_row = {
        "timestamp": now_dt.isoformat(),
        "net_assets": round(net_now, 2),
        "principal": round(init_cap, 2),
        "total_assets": round(total_assets_now, 2)
    }
    df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
    
    # 清理临时列 dt，保持 csv 干净
    save_df = df_hist.drop(columns=["dt"], errors="ignore")
    save_df.to_csv(HISTORY_FILE, index=False)

    # 绘图
    has_chart = plot_performance_chart(df_hist)

    # 调仓逻辑计算
    leverage_triggered = abs(leverage - cfg["target_leverage"]) > cfg["leverage_tolerance"]
    actions = []
    for code, s in asset_states.items():
        curr_w = s["cny_val"] / total_assets_now
        diff_w = curr_w - s["target_w"]
        s["curr_w"] = curr_w
        s["diff_w"] = diff_w

        target_cny = total_assets_now * s["target_w"]
        if s["currency"] == "CNY":
            target_local = target_cny
        elif s["currency"] == "USDT":
            target_local = target_cny / (usdt_usd_now * usd_cny_now)
        elif s["currency"] == "USDC":
            target_local = target_cny / (usdc_usd_now * usd_cny_now)

        diff_local = target_local - s["local_val"]
        s["target_local"] = target_local
        s["diff_local"] = diff_local

        rel_diff = diff_w / s["target_w"]
        if abs(rel_diff) > cfg["weight_rel_tolerance"] or leverage_triggered:
            raw_qty = diff_local / s["local_price"]
            suggested_qty = math.trunc(raw_qty / 100) * 100 if s["currency"] == "CNY" else round(raw_qty, 3)
            actions.append({
                "name": s["name"],
                "group": s["group"],
                "currency": s["currency"],
                "curr_w": curr_w,
                "target_w": s["target_w"],
                "curr_local": s["local_val"],
                "target_local": target_local,
                "diff_local": diff_local,
                "suggested_qty": suggested_qty
            })

    return {
        "repo": cfg.get("github_repo", ""),
        "total_assets": total_assets_now,
        "net_assets": net_now,
        "leverage": leverage,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "day_pnl": day_pnl,
        "week_pnl": week_pnl,
        "month_pnl": month_pnl,
        "year_pnl": year_pnl,
        "asset_states": asset_states,
        "actions": actions,
        "leverage_triggered": leverage_triggered,
        "has_chart": has_chart
    }

def send_notification(d):
    token = os.environ.get("PUSHPLUS_TOKEN")
    if not token:
        print("错误: 未配置 PUSHPLUS_TOKEN")
        return

    has_actions = len(d["actions"]) > 0 or d["leverage_triggered"]
    status_tag = "⚠️【调仓提醒】" if has_actions else "📊"
    title = f"{d['day_pnl'][0]:+,.0f} | {d['net_assets']:,.0f}"

    lines = [
        f"### {status_tag} 组合状态概览",
        f"- **总资产**: ¥{d['total_assets']:,.0f} | **净资产**: ¥{d['net_assets']:,.0f}",
        f"- **实际杠杆**: **{d['leverage']:.4f}x** (目标: 1.50x)",
        f"- **今日盈亏**: {format_pnl_str(d['day_pnl'][0], d['day_pnl'][1])}",
        f"- **本周盈亏**: {format_pnl_str(d['week_pnl'][0], d['week_pnl'][1])}",
        f"- **本月盈亏**: {format_pnl_str(d['month_pnl'][0], d['month_pnl'][1])}",
        f"- **本年盈亏**: {format_pnl_str(d['year_pnl'][0], d['year_pnl'][1])}",
        f"- **总累计盈亏**: {format_pnl_str(d['total_pnl'], d['total_pnl_pct'])}\n",
    ]

    # 如果有折线图，通过 jsDelivr CDN 嵌入图片
    if d["has_chart"] and d["repo"]:
        t_stamp = int(datetime.now().timestamp())
        # 使用 github raw 链接（要求仓库是 Public）
        chart_url = f"https://raw.githubusercontent.com/{d['repo']}/main/{CHART_FILE}?t={t_stamp}"
        lines.append(f"### 📈 净资产走势\n\n![资产走势]({chart_url})\n")

    lines.append("### 📦 持仓分布")
    for code, s in d["asset_states"].items():
        diff_str = format_diff(s["diff_w"] * 100)
        base_line = f"- **{s['name']}**: ¥{s['cny_val']:,.0f} ({s['target_w']*100:.2f}% {diff_str})"
        if s["currency"] != "CNY":
            base_line += f" [{s['local_val']:,.2f} {s['currency']}]"
        lines.append(base_line)

    if has_actions:
        lines.append("\n### 🚨 调仓操作指南 (对齐目标市值即可)")
        for act in d["actions"]:
            action_type = "买入" if act["diff_local"] > 0 else "卖出"
            curr = act["currency"]
            lines.append(f"**【{act['group']}】{act['name']}** (当前 {act['curr_w']*100:.2f}% ➡️ 目标 {act['target_w']*100:.2f}%)")
            lines.append(f"- 目标市值: **{act['target_local']:,.2f} {curr}**（现值 {act['curr_local']:,.2f} {curr}）")
            lines.append(f"- 建议操作: **{action_type} 约 {abs(act['diff_local']):,.2f} {curr}** (约 {abs(act['suggested_qty'])} 股/个)\n")
    else:
        lines.append("\n✅ **无需调仓**：所有资产与杠杆偏离均在安全区间内。")

    requests.post("http://www.pushplus.plus/send", json={
        "token": token,
        "title": title,
        "content": "\n".join(lines),
        "template": "markdown"
    })

if __name__ == "__main__":
    data = evaluate()
    send_notification(data)
