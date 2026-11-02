# توجيه Tawjeeh – نظام المواقف الذكية

يقرأ فيديو كاميرا مواقف، ويحدد المواقف المتاحة (أخضر) والمشغولة (أحمر) وغير المؤكدة (أصفر)، مع عدّاد مباشر، ويحفظ فيديو MP4 ناتجًا.
يعمل على CPU فقط، ولا يحتاج PyTorch أو بطاقة رسومية.

## التثبيت (مرة واحدة)

يتطلب Git وPython 3.13. افتح PowerShell في المكان اللي تبي تحفظ فيه المشروع، ثم شغّل:

```powershell
git clone https://github.com/mohmed-tech1/tawjeeh.git
cd tawjeeh
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## التشغيل

```powershell
.\.venv\Scripts\python.exe main.py                  # كل الفيديوهات في المجلد وفي videos\
.\.venv\Scripts\python.exe main.py Parking.mp4      # فيديو واحد
.\.venv\Scripts\python.exe main.py --no-show        # بدون نافذة عرض (أسرع)
.\.venv\Scripts\python.exe main.py --detect-every 2 # كشف كل إطارين (للأجهزة الضعيفة)
```

اضغط `Q` لإيقاف نافذة العرض.

**الناتج** في مجلد `output\`:
- `<اسم>_result.mp4`: الفيديو المعالَج.
- `<اسم>_counts.csv`: العدادات كل نصف ثانية.
- `<اسم>_summary.json`: الحالة النهائية لكل موقف.

## إضافة فيديو جديد

ضع الملف في مجلد المشروع أو في `videos\` ثم شغّل `main.py`. في أول مرة:
1. يُكتشف مكان كل موقف تلقائيًا، وتُحفظ الإعدادات في `configs\<اسم>.json`.
2. راجع الصورة `configs\<اسم>_spots_preview.jpg` للتحقق.

لكل كاميرا إعداداتها الخاصة. إذا اختلفت الكاميرا أو الزاوية عن ملف الإعدادات، يرفض البرنامج استخدامه.
لفيديو آخر من **نفس الكاميرا** استخدم: `main.py video2.mp4 --config configs\Parking.json`

## المعايرة اليدوية (عند فشل الاكتشاف التلقائي أو لتصحيحه)

```powershell
.\.venv\Scripts\python.exe calibrate.py Parking.mp4          # مراجعة وتعديل المواقف المكتشفة
.\.venv\Scripts\python.exe calibrate.py cam2.mp4 --empty     # تحديد يدوي كامل من الصفر
```

| الإجراء | الوظيفة |
|---|---|
| نقرة يسار ×4 | إضافة موقف (4 زوايا) |
| نقرة يسار على `?` رمادي | قبول مكان مرشح |
| نقرة يمين | حذف موقف |
| `D` | وضع النسخ: كل نقرة تضع نسخة من آخر موقف |
| `C` | قبول كل الأماكن المرشحة |
| `Z` | تراجع |
| `S` | حفظ |
| `Q` | خروج |

## الملفات

| الملف | الوظيفة |
|---|---|
| `main.py` | التشغيل الرئيسي |
| `calibrate.py` | أداة المعايرة اليدوية |
| `tawjeeh\detector.py` | كشف السيارات (YOLO ONNX) |
| `tawjeeh\stabilizer.py` | تثبيت الصورة عند اهتزاز الكاميرا أو حركة الدرون |
| `tawjeeh\spots.py` | الاكتشاف التلقائي للمواقف وحفظ الإعدادات |
| `tawjeeh\occupancy.py` | تحديد الحالة والتثبيت الزمني |
| `tawjeeh\render.py` | الرسم والعدادات |
| `tawjeeh\pipeline.py` | ربط المراحل |
| `models\` | يحتوي `yolo11n-obb.onnx` (للتصوير من الأعلى) و`yolo11n.onnx` (للكاميرات الأرضية) |
