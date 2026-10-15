# การออกแบบกลยุทธ์ Stochastic + Order Block (XAUUSD M30)

เอกสารนี้คือ spec ที่โค้ด `stoch_ob.py` ต้องทำตามทุกข้อ และใช้เป็นเช็กลิสต์ให้ agent ตรวจโค้ด

## 1. กติกาจากผู้ใช้ และการตีความเป็นเงื่อนไขในโค้ด

| กติกาของผู้ใช้ | การตีความในโค้ด (ค่าเริ่มต้น) | ตัวเลือกที่ปรับได้ |
|---|---|---|
| Stoch อยู่ 80–100 → SELL, อยู่ 0–20 → BUY | โซนบน `K ≥ upper` (80) · โซนล่าง `K ≤ lower` (20) | `upper`, `lower` |
| ออกไม้ (เปิดไม้) เมื่อเส้น Stochastic กลับทิศก่อน | `entry_mode="hook"`: แท่งก่อนหน้า K อยู่ในโซน และ K แท่งนี้กลับทิศ (SELL: `K[t] < K[t-1]`, BUY: `K[t] > K[t-1]`) | `"cross"` = K ตัด D ขณะอยู่ในโซน · `"exit_zone"` = K ออกจากโซน (ตัดลง 80 / ตัดขึ้น 20) |
| TP = OB ที่ใกล้สุดในทิศทางนั้น **หรือ** เมื่อ Stoch ถึงโซนฝั่งตรงข้าม | `tp_mode="first"`: ปิดเมื่อเกิดอย่างใดอย่างหนึ่งก่อน · ราคาแตะ OB (SELL: OB ฝั่ง demand ใต้ราคา) หรือ K เข้าโซนตรงข้าม (SELL: `K ≤ lower`) ปิดที่ราคาปิดแท่งนั้น · ถ้าไม่มี OB ในทิศนั้น ใช้ Stoch อย่างเดียว | `"ob_only"` (ไม่ใช้ Stoch ออก ยกเว้นไม่มี OB) · `"stoch_only"` (ไม่ใช้ OB) |
| SL ใต้ (เหนือ) Swing ก่อนหน้าที่เข้าไม้ | SELL: SL = high สูงสุดของ `sl_lookback` แท่งล่าสุด (รวมแท่งเข้า) + buffer · BUY: low ต่ำสุด − buffer · buffer = spread | `sl_lookback` · `sl_mode="fractal"` = swing ที่ยืนยันแล้ว (pivot) ล่าสุดที่อยู่เหนือ (ใต้) ราคาเข้า |
| ค่า parameter ของ Stoch ปรับได้ | `k_period=14, d_period=3, slowing=3` | ทั้งหมดเป็น input |
| ใช้ TF M30 ก่อน | M30 จาก M15 resample | |

**กติกาที่ผมเพิ่ม** (ผู้ใช้ไม่ได้ระบุ ทุกข้อปิดได้):
- `min_rr = 1.0`: ถ้า OB ที่ใกล้สุดให้ระยะ TP น้อยกว่า 1R ให้**ข้ามเทรด** (`ob_too_close="skip"`) เพราะ TP สั้นมากเมื่อหัก spread จะเสียเปรียบ · ปิดได้ด้วย `min_rr=0` · ตัวเลือก `"next"` = ใช้ OB ถัดไปที่ไกลพอ
- `one_per_zone = True`: เข้าได้ 1 ไม้ต่อการเข้าโซน 1 รอบ (กันการเข้าซ้ำหลังโดน SL ในโซนเดิม)
- ถือได้ทีละ 1 ไม้ (ไม่มีการเปิดซ้อน) เหมือน EA ทั่วไป
- `min_risk_atr = 0`: ตัวกรอง SL สั้นเกิน (ปิดไว้เป็นค่าเริ่มต้น)

## 2. นิยามตัวชี้วัด

### 2.1 Stochastic แบบ MT5 (MODE_SMA, STO_LOWHIGH)
```
LL[t] = min(low[t-k+1 .. t]),  HH[t] = max(high[t-k+1 .. t])         k = k_period
K[t]  = 100 * Σ(close − LL)[t-s+1 .. t] / Σ(HH − LL)[t-s+1 .. t]     s = slowing
        (ถ้าตัวหาร = 0 → K = 100 ตาม MT5)
D[t]  = SMA(K, d_period)
```
ค่าแท่งที่ข้อมูลไม่พอ = NaN (ไม่มีสัญญาณ)

### 2.2 Swing (fractal pivot)
pivot high ที่แท่ง i: `high[i] = max(high[i-p .. i+p])` และ `high[i] > max(high[i-p .. i-1])` (p = `ob_pivot`, ค่าเริ่ม 3)
**รู้ได้เมื่อแท่ง i+p ปิดแล้วเท่านั้น** (ไม่มี look-ahead) · pivot low กลับด้าน

### 2.3 Order Block
- **Bullish OB (demand)**: เมื่อแท่ง t ปิดเหนือ swing high ล่าสุดที่ยังไม่ถูกทะลุ (structure break ขึ้น)
  หาแท่ง low ต่ำสุด (`lo_bar`) ระหว่าง swing high นั้นถึง t แล้วไล่ย้อนจาก `lo_bar` ถอยหลังไม่เกิน 5 แท่ง
  เพื่อหาแท่งแดง (`close < open`) แท่งแรก = OB · โซน = [low, high] ของแท่งนั้น · ใช้ได้ตั้งแต่แท่ง t
- **Bearish OB (supply)**: กลับด้าน (ปิดใต้ swing low · แท่ง high สูงสุด · แท่งเขียวก่อนหน้า)
- **Fresh**: ยังไม่ถูกราคาแตะหลังสร้าง (bullish: แตะเมื่อ `low ≤ top` ที่แท่งหลังแท่งสร้าง · bearish: `high ≥ bottom`)
  OB ที่ถูกแตะแล้วไม่นำมาใช้เป็น TP
- **OB สำหรับ TP**:
  SELL → bullish OB ที่ fresh และ `top < entry` ตัวที่ top สูงสุด (ใกล้สุด) · TP = top + spread (ปิดก่อนถึงโซน)
  BUY → bearish OB ที่ fresh และ `bottom > entry` ตัวที่ bottom ต่ำสุด · TP = bottom − spread
- OB ใช้ข้อมูล ณ แท่งเข้าเท่านั้น TP คงที่ตลอดอายุไม้

## 3. ลำดับการทำงานต่อแท่ง (event loop, ทุกอย่างคำนวณ ณ ราคาปิดแท่ง t)
1. อัปเดตสถานะ fresh ของ OB ด้วย high/low แท่ง t
2. ยืนยัน pivot ที่แท่ง t − p
3. ตรวจ structure break ด้วย close[t] → สร้าง OB ใหม่
4. ถ้ามีไม้เปิดอยู่: จัดการไม้ (ข้อ 4) แล้วข้ามไปแท่งถัดไป
5. ถ้าไม่มีไม้: ตรวจสัญญาณ (ข้อ 1) → คำนวณ SL, risk, TP → ถ้าผ่านเงื่อนไข เปิดไม้ที่ close[t]

## 4. การจัดการไม้ (เริ่มจากแท่ง t+1)
ทุกแท่ง j ตามลำดับนี้ (เข้าข้างอนุรักษ์นิยม):
1. **SL**: SELL โดนเมื่อ `high[j] ≥ SL` ปิดที่ `max(open[j], SL)` (ถ้าเปิด gap ผ่าน SL ปิดที่ open) · BUY กลับด้าน
2. **TP (OB)**: SELL ถึงเมื่อ `low[j] ≤ TP` ปิดที่ TP · ถ้าแท่งเดียวโดนทั้ง SL และ TP **นับเป็น SL**
3. **Stoch ถึงโซนตรงข้าม** (ประเมินที่ราคาปิด): SELL `K[j] ≤ lower` → ปิดที่ close[j] · BUY `K[j] ≥ upper`
4. ข้อมูลหมด → ปิดที่ close สุดท้าย (EOD)
- แท่งที่ปิดไม้แล้ว ไม่เปิดไม้ใหม่ในแท่งเดียวกัน (เปิดได้ตั้งแต่แท่งถัดไป)

## 5. ต้นทุนและการวัดผล
- ราคาในข้อมูลเป็น bid · ต้นทุน spread ต่อไม้ = `spread` (2012–2022: 0.30 USD, 2026: 0.50 USD)
- `R = side × (exit − entry) / risk − spread / risk` โดย `side = −1` (SELL) / `+1` (BUY), `risk = |entry − SL|`
- ตัวชี้วัด: จำนวนไม้, ไม้ต่อปี, win %, avg R, PF, Max DD (R), t-stat, สัดส่วนเหตุผลการปิด (SL / TP_OB / STOCH / EOD), เวลาถือเฉลี่ย

## 6. ข้อมูลและการแบ่งช่วงทดสอบ
- XAUUSD M15 → M30: 2012-05 → 2022-03 (เวลา server = NY+7) และ 2026-03 → 2026-09 (UTC)
- **In-sample** 2012–2017 (ใช้เลือกค่า) · **Out-of-sample** 2018–2022 และ 2026 (ใช้ตัดสิน)

## 7. แผนการทดสอบ
1. **ค่าเริ่มต้น** (ตามกติกาผู้ใช้ตรงที่สุด) ทั้ง 3 ช่วง
2. **Sensitivity ทีละตัว** (เปลี่ยนค่าเดียว คงที่ค่าอื่น):
   Stoch (5,3,3) / (14,3,3) / (21,5,5) · โซน 80/20 vs 90/10 · entry_mode hook / cross / exit_zone ·
   sl_lookback 5 / 10 / 20 · min_rr 0 / 1 / 2 · tp_mode first / ob_only / stoch_only
3. เลือกแบบจากช่วง in-sample เท่านั้น แล้วรายงานผล out-of-sample ตรงไปตรงมา

## 8. การตรวจสอบโค้ดก่อน backtest
- Unit tests (`test_stoch_ob.py`): สูตร Stochastic เทียบการคำนวณมือ, pivot ไม่มี look-ahead,
  OB สร้างถูกแท่ง, ลำดับ SL ก่อน TP, gap fill, สูตร R, สัญญาณ hook/cross/exit_zone
- Agent ตรวจโค้ดแยก: อ่าน spec นี้และโค้ดทั้งหมด หาจุดที่ไม่ตรง spec, look-ahead, off-by-one,
  ความสมมาตรของฝั่ง BUY/SELL แล้วรายงานก่อนรัน backtest
