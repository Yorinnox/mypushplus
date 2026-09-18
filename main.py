import os
import json
import math
import requests
import pandas as pd
import yfinance as yf

# 颜色渲染工具（国内红涨绿跌习惯）
def color_text(val, text):
    if val > 0:
        return f'<font color="#d93025">{text}</font>'  # 红色
    elif val < 0:
        return f'<font color="#188038">{text}</font>'  # 绿色
    return text

def format_diff(diff_pct):
    sign = "+" if diff_pct > 0 else ""
    raw_str = f"{sign}{diff_pct:.2f}%"
    return color_text(diff_pct, raw_str)

def format_pnl_str(pnl_val, pct_val):
    sign = "+" if pnl_val > 0 else ""
    raw_str = f"{sign}¥{pnl_val:,.0f} ({sign}{pct_val:.2f}%)"
    return color_text(pnl_val, raw_str)

def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_market_history(assets):
    tickers = [item["ticker"] for item in assets.values()]
    tickers += ["USDCNY=X", "USDT-USD", "USDC-USD"]
    
    # 智能代理识别：如果在 GitHub Actions 云端运行，则直连（无需代理）；若在本地运行，自动走 Clash 7890 端口
    # proxy = None if os.environ.get("GITHUB_ACTIONS") else "http://127.0.0.1:7890"
    proxy = None

    # 获取近 1 个月日线
    df = yf.download(tickers=tickers, period="1mo", interval="1d", progress=False, proxy=proxy)['Close']
    
    # 核心修复 1：ffill() 让休市资产顺延周五收盘价，bfill() 补齐开头空缺，彻底替代有隐患的 dropna()
    df = df.ffill().bfill()

    # 核心修复 2：按真实自然日进行跨市场日历对齐（解决周末多出 2 根币圈 K 线导致的 iloc 错位问题）
    latest = df.iloc[-1]
    latest_dt = df.index[-1]

    dt_24h = latest_dt - pd.Timedelta(days=1)
    dt_7d = latest_dt - pd.Timedelta(days=7)

    # asof 会精准取到 <= 对应时间戳的历史行情
    prev_24h = df.asof(dt_24h)
    prev_7d = df.asof(dt_7d)

    # 极端情况兜底
    if prev_24h is None or (hasattr(prev_24h, 'isna') and prev_24h.isna().any()):
        prev_24h = df.iloc[-2] if len(df) >= 2 else latest
    if prev_7d is None or (hasattr(prev_7d, 'isna') and prev_7d.isna().any()):
        prev_7d = df.iloc[0]

    return {
        "df": df,
        "latest": latest,
        "prev_24h": prev_24h,
        "prev_7d": prev_7d
    }

def evaluate():
    cfg = load_config()
    assets = cfg["assets"]
    hist = fetch_market_history(assets)

    latest = hist["latest"]
    prev_24h = hist["prev_24h"]
    prev_7d = hist["prev_7d"]

    # 汇率处理（周末外汇休市时，自动顺延周五汇率）
    usd_cny_now = float(latest["USDCNY=X"])
    usdt_usd_now = float(latest["USDT-USD"])
    usdc_usd_now = float(latest["USDC-USD"])

    usd_cny_24h = float(prev_24h["USDCNY=X"])
    usdt_usd_24h = float(prev_24h["USDT-USD"])
    usdc_usd_24h = float(prev_24h["USDC-USD"])

    usd_cny_7d = float(prev_7d["USDCNY=X"])
    usdt_usd_7d = float(prev_7d["USDT-USD"])
    usdc_usd_7d = float(prev_7d["USDC-USD"])

    total_assets_now = 0.0
    total_assets_24h = 0.0
    total_assets_7d = 0.0
    asset_states = {}

    for code, info in assets.items():
        ticker = info["ticker"]
        qty = info["qty"]
        curr_type = info["currency"]

        p_now = float(latest[ticker])
        p_24h = float(prev_24h[ticker])
        p_7d = float(prev_7d[ticker])

        # 本币单价与人民币折算
        if curr_type == "CNY":
            local_p_now = p_now
            cny_val_now = qty * p_now
            cny_val_24h = qty * p_24h
            cny_val_7d = qty * p_7d
        elif curr_type == "USDT":
            local_p_now = p_now / usdt_usd_now
            cny_val_now = qty * p_now * usd_cny_now
            cny_val_24h = qty * p_24h * usd_cny_24h
            cny_val_7d = qty * p_7d * usd_cny_7d
        elif curr_type == "USDC":
            local_p_now = p_now / usdc_usd_now
            cny_val_now = qty * p_now * usd_cny_now
            cny_val_24h = qty * p_24h * usd_cny_24h
            cny_val_7d = qty * p_7d * usd_cny_7d

        total_assets_now += cny_val_now
        total_assets_24h += cny_val_24h
        total_assets_7d += cny_val_7d

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

    # 宏观杠杆与盈亏
    debt = cfg["debt_cny"]
    net_now = total_assets_now - debt
    net_24h = total_assets_24h - debt
    net_7d = total_assets_7d - debt

    leverage = total_assets_now / net_now if net_now > 0 else 0
    init_cap = cfg["initial_capital_cny"]

    total_pnl = net_now - init_cap
    total_pnl_pct = (total_pnl / init_cap) * 100

    pnl_24h = net_now - net_24h
    pnl_24h_pct = (pnl_24h / net_24h) * 100 if net_24h > 0 else 0

    pnl_7d = net_now - net_7d
    pnl_7d_pct = (pnl_7d / net_7d) * 100 if net_7d > 0 else 0

    leverage_triggered = abs(leverage - cfg["target_leverage"]) > cfg["leverage_tolerance"]

    # 计算目标市值与再平衡操作
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

        # 判断偏离阈值
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
        "rates": {"USDCNY": usd_cny_now},
        "total_assets": total_assets_now,
        "net_assets": net_now,
        "leverage": leverage,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "pnl_24h": pnl_24h,
        "pnl_24h_pct": pnl_24h_pct,
        "pnl_7d": pnl_7d,
        "pnl_7d_pct": pnl_7d_pct,
        "asset_states": asset_states,
        "actions": actions,
        "leverage_triggered": leverage_triggered
    }

def send_notification(d):
    # 优先使用环境变量，本地兜底使用你的默认 Token
    token = os.environ.get("PUSHPLUS_TOKEN")
    if not token:
        print("错误: 未配置 PUSHPLUS_TOKEN")
        return

    has_actions = len(d["actions"]) > 0 or d["leverage_triggered"]
    status_tag = "⚠️【调仓提醒】" if has_actions else "📊【组合巡检】"
    # 保留你修改的 4 位小数杠杆格式
    title = f"{status_tag} 杠杆: {d['leverage']:.4f}x | 24h: {d['pnl_24h_pct']:+.2f}%"

    lines = [
        "### 📈 组合状态概览",
        f"- **总资产**: ¥{d['total_assets']:,.0f} | **净资产**: ¥{d['net_assets']:,.0f}",
        f"- **实际杠杆**: **{d['leverage']:.4f}x** (目标: 1.50x)",
        f"- **近 24 小时盈亏**: {format_pnl_str(d['pnl_24h'], d['pnl_24h_pct'])}",
        f"- **近 7 天累计盈亏**: {format_pnl_str(d['pnl_7d'], d['pnl_7d_pct'])}",
        f"- **成立以来总盈亏**: {format_pnl_str(d['total_pnl'], d['total_pnl_pct'])}\n",
        "### 📦 持仓分布"
    ]

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
