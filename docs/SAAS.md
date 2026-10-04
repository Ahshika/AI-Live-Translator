# الحسابات والاشتراكات (المرحلتان ١٥ و١٧): تصميم جاهز للتنفيذ

> **الحالة:** تصميم فقط، عن قصد. البرنامج الحالي يعمل بالكامل على جهاز المستخدم بلا خادم وبلا تكلفة.
> الحسابات والاشتراكات تحتاج خادمًا على الإنترنت، أي تكلفة شهرية ودومين وبوابة دفع،
> لذلك تُنفَّذ عندما تتوفر ميزانية. صُمم الكود الحالي بحيث لا يتغير: الخادم **يضيف** خدمات ولا يستبدل المحرك.

## لماذا يبقى المحرك محليًا حتى في نسخة الـ SaaS؟
- **التكلفة:** ترجمة الصوت على خوادم GPU سحابية هي أغلى جزء في أي منتج من هذا النوع. هنا يتحملها جهاز المستخدم.
- **الخصوصية:** صوت المكالمة لا يغادر الجهاز. هذه ميزة بيع قوية للشركات.
- **السرعة:** لا يوجد ذهاب وعودة للإنترنت.

إذن الخادم يدير فقط: الحسابات، والترخيص، والدقائق المستهلكة، ومزامنة الإعدادات والسجل (اختياريًا).

## المعمارية

```
Desktop app (المحرك المحلي)
   │  HTTPS + توكن قصير العمر (JWT ١٥ دقيقة + refresh token)
   ▼
API Server (FastAPI) ── PostgreSQL
   ├── /auth      تسجيل / دخول / Google OAuth / نسيت كلمة المرور
   ├── /license   تفعيل الجهاز + الخطة الحالية + الحد الشهري
   ├── /usage     رفع دقائق الاستخدام (أرقام فقط، بدون نص أو صوت)
   ├── /settings  مزامنة الإعدادات بين الأجهزة
   └── /history   مزامنة السجل (اختياري، مشفّر من طرف إلى طرف)
```

## قاعدة البيانات (PostgreSQL)

```sql
CREATE TABLE users (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email         CITEXT UNIQUE NOT NULL,
  password_hash TEXT,                       -- argon2id; NULL for Google-only accounts
  google_sub    TEXT UNIQUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE subscriptions (
  user_id     UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  plan        TEXT NOT NULL CHECK (plan IN ('free', 'pro', 'business')),
  minutes_cap INT,                          -- NULL = unlimited
  renews_at   TIMESTAMPTZ,
  provider_ref TEXT                         -- payment gateway subscription id
);

CREATE TABLE devices (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name        TEXT,
  last_seen   TIMESTAMPTZ
);

CREATE TABLE sessions (                     -- one translation session (a call)
  id              UUID PRIMARY KEY,
  user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  device_id       UUID REFERENCES devices(id),
  source_language TEXT NOT NULL,
  target_language TEXT NOT NULL,
  platform        TEXT,                     -- zoom / teams / system ...
  started_at      TIMESTAMPTZ NOT NULL,
  ended_at        TIMESTAMPTZ
);

CREATE TABLE usage (                        -- numbers only: never text or audio
  session_id  UUID REFERENCES sessions(id) ON DELETE CASCADE,
  day         DATE NOT NULL,
  speech_sec  INT NOT NULL DEFAULT 0,       -- seconds of speech recognised
  chars_mt    INT NOT NULL DEFAULT 0,       -- characters translated
  speech_out_sec INT NOT NULL DEFAULT 0,    -- seconds of voice generated
  PRIMARY KEY (session_id, day)
);

CREATE TABLE user_settings (
  user_id  UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  data     JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE messages (                     -- only if the user enables history sync
  id              BIGSERIAL PRIMARY KEY,
  session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  speaker         TEXT NOT NULL CHECK (speaker IN ('me', 'other')),
  ciphertext      BYTEA NOT NULL,           -- encrypted on the device; server can't read it
  created_at      TIMESTAMPTZ NOT NULL
);
```

السجل المحلي الحالي (`app/engine/history.py`، SQLite) يستخدم نفس الحقول، فالمزامنة لاحقًا هي مجرد رفع للصفوف.

## الخطط (مثال من طلبك)

| الخطة | الدقائق/شهر |
|---|---|
| Free | 100 |
| Pro | 1000 |
| Business | غير محدود أو حسب العقد |

**كيف تُحسب الدقيقة:** ثواني الكلام الفعلي التي يكتشفها كاشف الكلام في الاتجاهين، وليست مدة المكالمة. هذا أعدل للمستخدم. المحرك يجمعها محليًا ويرفعها كل ٥ دقائق. وإذا انقطع الإنترنت يكمل عملًا بآخر رصيد معروف (مهلة سماح ٧ أيام).

## الأمان
- **كلمات المرور:** argon2id. **التوكنات:** JWT قصير العمر + refresh token يدور مع كل استخدام. وفي سطح المكتب يُخزَّن الـ refresh token في **Windows Credential Manager**، لا في ملف.
- **الاتصال:** HTTPS/WSS فقط. حد معدل للطلبات على /auth (مثلًا ٥ محاولات في الدقيقة لكل IP ولكل بريد).
- **الدفع:** لا تمر بيانات بطاقة على خادمنا أبدًا، بل عبر صفحة بوابة الدفع وwebhooks موقّعة.
- **المفاتيح:** لا توجد مفاتيح API في الواجهة. والمحرك المحلي لا يحتاج مفاتيح أصلًا.

## خيارات استضافة منخفضة التكلفة (تحقق من الأسعار وقت التنفيذ)
- خادم API صغير (1 vCPU / 1GB) يكفي آلاف المستخدمين، لأن الخادم لا يعالج صوتًا.
- PostgreSQL مُدار بخطة صغيرة، أو على نفس الخادم في البداية.
- المحرك الثقيل يبقى على أجهزة المستخدمين، فتكلفة الخادم لا تكبر مع دقائق الترجمة.

## ⚠️ قبل البيع: التراخيص

| المكوّن | الترخيص | أثره على النسخة المدفوعة |
|---|---|---|
| Whisper (faster-whisper) | MIT | ✅ مسموح |
| **NLLB-200** (الترجمة) | **CC-BY-NC 4.0** | ❌ **غير تجاري**: يجب استبداله، مثلًا بـ MADLAD-400 (Apache-2.0) أو Opus-MT أو خدمة مدفوعة. الاستبدال مجرد provider جديد |
| Piper (المكتبة) | GPL-3.0 | ⚠️ يحتاج مراجعة قانونية عند توزيعه داخل منتج مغلق المصدر، أو تشغيله كعملية منفصلة |
| أصوات Piper | لكل صوت ترخيصه | ⚠️ راجع ملف MODEL_CARD لكل صوت |
| VB-CABLE | Donationware | ⚠️ لا يُدمج داخل المثبّت بدون ترخيص توزيع من VB-Audio. حاليًا يثبّته المستخدم بنفسه |
