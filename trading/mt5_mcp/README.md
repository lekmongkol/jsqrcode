# ติดตั้ง MT5 MCP ให้ Claude สั่งเปิด/ปิดออร์เดอร์ (บัญชี Demo)

ใช้ **[metatrader-mcp-server](https://github.com/ariadng/metatrader-mcp-server) v0.5.1** (MIT, ผู้พัฒนา Aria Dhanang)
· ตรวจ source code v0.5.1 แล้ว (15 ต.ค. 2026): รหัสผ่านถูกส่งให้ MT5 ของเครื่องคุณเท่านั้น ไม่มีการส่งข้อมูลออกไปที่อื่น

> ต้องทำบน **เครื่อง Windows ที่เปิด MT5 อยู่** · Claude ใน cloud session ต่อกับ MT5 บนเครื่องคุณโดยตรงไม่ได้
> ต้องใช้ **Claude Desktop** (หรือ Claude Code) ที่ติดตั้งบนเครื่องเดียวกับ MT5

## 1. เตรียม MT5 (บัญชี Demo)
1. เปิด MT5 ที่ใช้อยู่ → `File` → `Open an Account` → เลือกโบรกเกอร์ → **Open a demo account**
   (หรือใช้บัญชี demo ที่มีอยู่) จด **Login, Password, Server** ไว้
2. `Tools` → `Options` → แท็บ `Expert Advisors` → ติ๊ก **Allow algorithmic trading** → `OK`
3. หาที่อยู่ไฟล์ `terminal64.exe` ของ MT5 ตัวนี้: คลิกขวาที่ไอคอน MT5 → `Open file location`
   (ปกติคือ `C:\Program Files\<ชื่อโบรกเกอร์> MetaTrader 5\terminal64.exe`)

> ⚠️ เมื่อ MCP เชื่อมต่อ มันจะ **login MT5 ตัวนั้นด้วยบัญชีที่ใส่ใน config** ถ้า MT5 ตัวนี้ใช้กับบัญชีจริงอยู่
> แนะนำให้ติดตั้ง MT5 อีกชุด (หรือโบรกเกอร์อื่น) สำหรับ demo แยก แล้วชี้ `--path` ไปที่ชุดนั้น

## 2. ติดตั้ง Python และ MCP server
1. ติดตั้ง [Python 3.10 ขึ้นไป](https://www.python.org/downloads/) (ติ๊ก **Add python.exe to PATH**)
2. เปิด Command Prompt แล้วรัน (ล็อกเวอร์ชันที่ตรวจแล้ว):
   ```bat
   pip install metatrader-mcp-server==0.5.1
   where metatrader-mcp-server
   ```
   จดที่อยู่ไฟล์ที่ได้จาก `where` (เช่น `C:\Users\<ชื่อ>\AppData\Local\Programs\Python\Python312\Scripts\metatrader-mcp-server.exe`)

## 3. ตั้งค่า Claude Desktop
เปิดไฟล์ `%APPDATA%\Claude\claude_desktop_config.json` (Claude Desktop → `Settings` → `Developer` → `Edit Config`) แล้วใส่:

```json
{
  "mcpServers": {
    "metatrader": {
      "command": "C:\\Users\\<ชื่อ>\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\metatrader-mcp-server.exe",
      "args": [
        "--login",     "<DEMO_LOGIN>",
        "--password",  "<DEMO_PASSWORD>",
        "--server",    "<DEMO_SERVER>",
        "--path",      "C:\\Program Files\\<โบรกเกอร์> MetaTrader 5\\terminal64.exe",
        "--transport", "stdio"
      ]
    }
  }
}
```

- **ต้องมี `"--transport", "stdio"` เสมอ** ถ้าลืม server จะเปิด port 8080 บนทุก network interface (0.0.0.0)
  **โดยไม่มีรหัสผ่าน** ใครที่อยู่ใน network เดียวกันก็สั่งเทรดบัญชีคุณได้
- ใน Windows path ต้องใช้ `\\` (backslash สองตัว)
- รหัสผ่านถูกเก็บในไฟล์นี้แบบไม่เข้ารหัส อย่าแชร์ไฟล์นี้ และใช้กับบัญชี demo เท่านั้นจนกว่าจะพิสูจน์กลยุทธ์ได้

ปิด Claude Desktop ให้หมด (รวมที่ system tray) แล้วเปิดใหม่

## 4. ทดสอบ (อ่านอย่างเดียวก่อน)
พิมพ์ใน Claude Desktop ทีละข้อ:
1. "ดูข้อมูลบัญชี MT5" → ตรวจว่า **trade mode เป็น DEMO** และ server ถูกต้อง
2. "ขอราคา XAUUSD ล่าสุด" (ชื่อ symbol อาจต่างตามโบรกเกอร์ เช่น `XAUUSD.m`, `GOLD`)
3. "ขอแท่ง H4 XAUUSD ล่าสุด 50 แท่ง"
4. ทดสอบสั่ง: "BUY XAUUSD 0.01 lot" → แล้ว "ตั้ง SL ของ position นั้นที่ ... " → "ปิด position นั้น"

## 5. ความปลอดภัยที่ต้องรู้
- **`place_market_order` ไม่มีช่อง SL/TP**: ออร์เดอร์ market จะเปิดโดย**ไม่มี SL** จนกว่าจะสั่ง `modify_position`
  ทุกครั้งที่เปิดไม้ ต้องให้ Claude ตั้ง SL ทันที (หรือใช้ `place_pending_order` ที่ใส่ SL/TP ได้ตั้งแต่แรก)
- มีคำสั่งอันตราย เช่น `close_all_positions` · `cancel_all_pending_orders` · ใน Claude Desktop
  **อย่ากด "Always allow"** ให้เครื่องมือสั่งเทรด ให้ Claude ขออนุญาตทุกครั้งที่จะเปิด/ปิดไม้
- ไม่มีตัวจำกัดความเสี่ยงในตัว (lot สูงสุด, ขาดทุนต่อวัน) → ตั้ง lot เล็กสุดเอง และใช้ demo
- ถ้าต้องการให้ระบบตัดสินใจตามกติกา backtest เป๊ะ (ไม่ให้ Claude ตีความเอง) และล็อก SL/lot ในโค้ด
  ขั้นต่อไปคือทำ MCP เฉพาะกลยุทธ์ที่ห่อเครื่องมือชุดนี้ไว้

## SHA256 ของไฟล์ที่ตรวจ (v0.5.1)
- wheel: `a5a8d0b36aad58b3c809504a90c17d66fc4ebc0c2ca9f401e030dc9f873f4734`
- sdist: `2f90944a21cc544d4cf56e92234175a2d1feac7afd35e0665ee93dc0c8b71b68`
