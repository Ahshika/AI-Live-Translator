# AI Live Translator — Architecture

> الوثيقة دي هي المرجع الأساسي للمشروع. أي معلومة عن أسعار أو قدرات API خارجية
> متعلّمة **"تحتاج تحقق"** لحد ما نراجعها من المصدر الرسمي في المرحلة اللي هنستخدمها فيها.

## 0. القيود اللي شكّلت التصميم

| القيد | أثره على التصميم |
|---|---|
| **مفيش ميزانية/كارت** | المسار الافتراضي **Local-first**: كل الـAI بيشتغل على الجهاز. الـCloud providers تبقى plugins اختيارية. |
| الجهاز: RTX 5060 8GB + 16GB RAM | يكفي Whisper (large-v3-turbo) + موديل ترجمة + TTS محلي في نفس الوقت. |
| Windows أولاً | الصوت عن طريق WASAPI، والمايك الوهمي عن طريق Virtual Audio Cable. |
| لازم يشتغل مع أي تطبيق اجتماعات | الأساس هو **Universal Mode** على مستوى نظام التشغيل، والـintegrations الخاصة بكل منصة تيجي بعد كده. |

## 1. الصورة الكاملة

```
┌──────────────────────────── Windows PC ─────────────────────────────┐
│                                                                     │
│  Real mic ──► [Outgoing Pipeline] ──► CABLE Input (VB-CABLE)        │
│   (Arabic)     VAD→STT→MT→TTS          │                            │
│                                        ▼                            │
│                              "CABLE Output" = Virtual Mic           │
│                                        │  (selected in Zoom/Discord)│
│                                        ▼                            │
│                               Meeting App ──► Internet ──► German   │
│                                        │                            │
│  Headphones ◄── [Incoming Pipeline] ◄──┘ meeting audio captured     │
│   (Arabic)      VAD→STT→MT→TTS          via WASAPI process loopback │
│                                                                     │
│  UI (Tauri/React, later) ◄── WebSocket ──► Engine (Python, local)   │
└─────────────────────────────────────────────────────────────────────┘
```

فيه **Pipeline** متطابقين ومستقلّين، كل واحد له اتجاه:

| | Outgoing (أنا ← هو) | Incoming (هو ← أنا) |
|---|---|---|
| Source | المايك الحقيقي | صوت تطبيق الاجتماع (loopback) |
| Language | لغتي (ar) | لغته (de أو Auto) |
| Sink | Virtual Mic (CABLE Input) | سماعاتي |
| Priority | عالية: الطرف التاني مستني | عالية: أنا مستني |

## 2. ليه Engine بلغة Python، وليه الـUI منفصل؟

* كل موديلات الـAI (Whisper, NLLB, Piper/XTTS) نسختها الأقوى في Python.
  لو نقلناها لـRust/Node هنعيد كتابة طبقة تكاملها كلها، ومن غير مكسب في الـlatency،
  لأن الشغل التقيل نفسه بيحصل جوه C++/CUDA.
* **Engine** = عملية Python بتشغّل FastAPI + WebSocket على `127.0.0.1` بس.
  الـUI بتكلّمه عن طريق WebSocket عشان تبعت الأوامر (Start/Stop/Settings) وتستقبل
  الـtranscripts والحالة.
* **Desktop UI**: الاختيار **Tauri + React**.
  * Electron بيحمّل Chromium كامل جوه البرنامج (~150MB+ RAM زيادة) ومش هيوفّر لنا أي ميزة في الصوت، لأن الصوت كله شغّال في الـEngine.
  * Tauri صغير، وفيه **Rust sidecar** نقدر نكتب فيه لاحقاً الأجزاء الـnative
    (process loopback capture، device notifications) لو Python ماكفّاش.
  * Python GUI (Qt) ممكن، لكن الواجهة هتبقى أضعف وأصعب في التطوير.
  * الـUI مش جزء من أول المراحل. هنبدأ بـCLI عشان نقيس الـpipeline نفسه.

## 3. Audio Architecture

**الصيغة الداخلية الموحّدة:** `PCM float32, mono, 16 kHz`
* Whisper ومعظم موديلات/خدمات الـSTT متدرّبة على 16kHz mono، فلو استخدمنا sample rate أعلى هندفع CPU زيادة ومش هنكسب أي دقة.
* بنحوّل من sample rate الجهاز (غالباً 48kHz) **مرة واحدة عند الإدخال**، وبنحوّل لـsample rate جهاز الإخراج **مرة واحدة عند الإخراج**.
* **Frame = 20ms (320 samples)**: ده المقاس اللي Silero/WebRTC VAD بيشتغلوا بيه، وصغير كفاية إنه ما يزوّدش latency ملحوظ.
* **Capture buffer:** callback من WASAPI (عن طريق sounddevice/PortAudio) → `queue` بحد أقصى معروف.
  لو الـconsumer بطّأ، بنرمي أقدم frames وبنسجّل "dropped audio" كـmetric، بدل ما الذاكرة تفضل تكبر.
* **Utterance buffer:** بيبدأ بـpre-roll حوالي 300ms عشان ما يقطعش أول حرف، وبيتقفل بعد صمت حوالي 500–700ms
  (Phase 6، والقيمة هتبقى قابلة للضبط). وفيه حد أقصى لطول الجملة (~15s) بنقطع عنده بالقوة.

## 4. AI Pipeline (لكل اتجاه)

```
frames ─► VAD ─► utterance ─► STT (partials + final)
                                   │ final only
                                   ▼
                     Segmenter (جمل كاملة / عبارات ذات معنى)
                                   ▼
                     Translation (src → tgt)
                                   ▼
                     TTS (streaming chunks) ─► Playback Queue ─► Sink
```

* **Partials للعرض بس:** الـpartial transcript بيظهر كنص على الشاشة،
  لكن **عمرنا ما بنترجم أو نولّد صوت من partial**. الترجمة بتحصل للـfinal segment بس (ده متطلب §5).
* **Segmenter:** بيقفل الـsegment عند صمت VAD، أو عند علامة نهاية جملة، أو لما يوصل للحد الأقصى للطول.
  ده بيمنع إننا نترجم كلمة كلمة.
* **Interfaces** (الكود في `backend/app/services/*/base.py`):
  * `SpeechToTextProvider.transcribe(audio, language|None) -> Transcript`
  * `TranslationProvider.translate(text, src, tgt) -> str`
  * `TextToSpeechProvider.synthesize_stream(text, lang, voice) -> AsyncIterator[pcm]`
  * أي provider (local أو cloud) بيتسجّل في registry، وبيتختار من الـconfig من غير ما نلمس باقي الكود.

### الـProviders المبدئية (Local، مجانية)

| المرحلة | الاختيار الأول | ليه | البدائل |
|---|---|---|---|
| STT | **faster-whisper** `large-v3-turbo` على GPU | مجاني، 99 لغة، أداء العربي فيه كويس، ومعاه language detection | `small` للأجهزة الضعيفة؛ Cloud: Deepgram/Azure/OpenAI (تحتاج تحقق من الأسعار) |
| Translation | **NLLB-200** distilled 1.3B (int8) عن طريق CTranslate2 | مجاني، 200 لغة، بيدعم **العامية المصرية** (`arz_Arab`)، سريع على GPU. ⚠️ الترخيص **CC-BY-NC** (غير تجاري) — لازم يتبدّل قبل أي نسخة مدفوعة | LLM محلي عن طريق Ollama (جودة أعلى في الكلام العامي لكنه أبطأ)؛ Cloud: DeepL/Google/Azure |
| TTS | **Piper** (سريع جداً، CPU) كبداية | مجاني، offline، فيه أصوات عربي (kareem) وألماني (thorsten/kerstin)، وبيشكّل العربي تلقائياً (tashkeel). الترخيص GPL-3.0 + ترخيص لكل صوت | XTTS-v2 (طبيعي أكتر، لكنه أتقل وترخيصه غير تجاري)؛ edge-tts (طبيعي لكن غير رسمي ⚠️) |

> ملاحظة جودة: اللهجة المصرية في STT والترجمة أضعف من الفصحى. هنقيس ده بأرقام في Test Plan
> بدل ما نفترضه.

## 5. Virtual Microphone (جوهر المشروع)

**المشكلة:** تطبيق الاجتماع بيقرأ من "مايك". إحنا عايزينه يقرأ الصوت **المترجَم**.

**الحل الواقعي:** نستخدم Virtual Audio Cable موجود ومُوقّع، تحديداً **VB-CABLE** (مجاني/donationware).
* بيعمل جهازين: `CABLE Input` (speaker) و `CABLE Output` (microphone).
* الـEngine بيشغّل صوت الـTTS على `CABLE Input`.
* في Zoom/Discord/Meet/Teams بتختار المايك `CABLE Output`.
* النتيجة: الطرف التاني بيسمع الألماني بس، ومش بيسمع صوتك العربي الأصلي أبداً.

**ليه مش نكتب Driver خاص بينا (يظهر باسم "AI Translator Mic")؟**
* لازم يبقى **kernel-mode audio driver** (AVStream/SysVAD)، وWindows ما بيقبلش driver غير موقّع.
  التوقيع محتاج **EV code-signing certificate** + **Microsoft attestation signing** (Partner Center)،
  وده بفلوس ومحتاج كيان قانوني. يعني مش مناسب دلوقتي.
* تصميمنا بيعزل ده ورا interface اسمه `AudioSink`، فلو اتحوّلنا لمنتج تجاري بعدين نقدر:
  1) نشتري ترخيص redistribution من VB-Audio، أو 2) نكتب driver موقّع بنفسنا، **من غير ما نغيّر باقي الكود**.

## 6. Capturing the Other Person's Audio

| الطريقة | بيلتقط إيه | ملاحظات |
|---|---|---|
| **WASAPI Loopback** (على جهاز الإخراج) | كل أصوات الويندوز | سهل وشغّال من Python عن طريق `PyAudioWPatch`، لكنه بيلتقط إشعارات ويوتيوب وأي حاجة تانية شغالة. |
| **WASAPI Process Loopback** (`AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`) | صوت **process معيّن** (Zoom.exe وأبناؤه) | ده الحل المطلوب. متاح في Windows 11 وإصدارات Windows 10 الحديثة. محتاج كود native (C++/Rust) → sidecar صغير بيبعت PCM للـEngine. |
| Meeting app → output = VB-CABLE B | صوت التطبيق ده بس | حل بديل من غير كود native، لكنه محتاج كابل افتراضي تاني وضبط يدوي. |

**الخطة:** Phase 8 تبدأ بالـLoopback العادي عشان يبقى أسهل، وبعدين نضيف Process Loopback.

**Echo/Loop:** الترجمة الصادرة بتروح لـCABLE بس، والترجمة الواردة بتروح للسماعات بس.
* ما ينفعش نلتقط الـloopback من الجهاز اللي بنشغّل عليه الترجمة الواردة. الـprocess loopback بيحل المشكلة دي تلقائياً، لأنه بيلتقط Zoom بس ومش بيلتقط الـEngine.
* السماعات (headphones) مطلوبة في الـMVP. لو المستخدم شغّال على speakers هنحتاج AEC، وده Phase لاحقة.

## 7. Interruptions & Playback Queue

* لكل Sink فيه `PlaybackQueue`: أقصى طول ليه 2 utterances.
* **Barge-in:** لو الـVAD لقى كلام جديد من نفس الشخص اللي بنترجم له، الـTTS اللي شغّال بيتعمل له fade-out حوالي 150ms.
  التشغيل ما بيتوقفش فوراً، والجملة الجديدة بتاخد الأولوية.
* لو الطابور اتملى، بنرمي أقدم جملة **ما بدأتش تشتغل لسه**، وبنعرض نصها كـsubtitle بس.
  كده عمرنا ما هنتراكم ونفضل ورا المحادثة بدقايق.

## 8. Latency: مصادرها وإزاي هنقيسها

```
capture(20ms) → VAD endpoint(~500ms silence) → STT → MT → TTS first chunk → playback buffer
```
* كل segment بياخد `trace_id` ومعاه timestamps لكل مرحلة (`t_speech_end`, `t_stt_done`, `t_mt_done`, `t_tts_first_audio`, `t_play_start`).
* الـmetric الأساسي: **end-of-speech → first translated audio**. هنسجّله في كل تجربة.
* أكبر عامل هو مدة الصمت اللي بنستناها عشان نقفل الجملة، وده tradeoff مباشر بين السرعة والدقة.
  هنعمل له Latency Mode (Fast/Balanced/Accurate).
* الموديلات بنحمّلها **مرة واحدة** وبتفضل على الـGPU. أي cloud provider بيستخدم persistent WebSocket.

## 9. Meeting Platforms: Universal vs Platform-specific

**Universal Mode (هو الـMVP):** Virtual Mic + Loopback. بيشتغل مع أي تطبيق بيستخدم مايك/سماعة،
ومن غير أي API أو حساب developer.

**Platform-specific (لاحقاً، وكل بند فيهم محتاج تحقق من الـdocs الرسمية الحالية قبل التنفيذ):**
* **Zoom:** Meeting SDK بيقدّم raw audio (mixed وper-user) وvirtual audio mic لـbot بيدخل كمشارك.
  RTMS بيبعت streams لـbackend، لكنه **استقبال بس**. الاتنين محتاجين Zoom app وموافقات.
* **Discord:** الـbots تقدر تدخل voice channel. استقبال الصوت مش مدعوم رسمياً بالكامل.
  أتمتة حساب المستخدم (selfbot) **مخالفة للشروط** ومش هنعملها. يعني Universal Mode هو الحل العملي.
* **Teams:** Real-time Media bots (C#/.NET على Windows Server) بيقدّموا per-participant audio. ده تعقيد كبير ومناسب لنسخة Business بس.
* **Google Meet:** Meet Media API (كان في developer preview)؛ وإلا Universal Mode.

## 10. Multi-speaker meetings
* لو عندنا per-user audio (Zoom SDK/Teams bot): نعمل pipeline لكل مشارك، وده الأدق.
* لو عندنا mixed audio بس: Diarization (pyannote) + language ID لكل segment.
  دقته بتقل مع تداخل الكلام والجمل القصيرة، وبيزوّد latency. هنعتبره ميزة "best-effort".

## 11. Privacy & Security (by design)
* **Default:** الصوت **ما بيتخزّنش أبداً**. الـchunks بتعيش في الذاكرة لحد ما الـsegment يخلص وبعدين بتتمسح.
* الـLocal mode معناه إن **ولا byte بيخرج من الجهاز**.
* الـConversation history: مقفول افتراضياً. لو اتفعّل، بيتحفظ **نص بس** على الجهاز.
* الـEngine بيسمع على `127.0.0.1` بس، ومعاه token عشوائي لكل تشغيل بيتبادله مع الـUI.
* أي Cloud keys بتبقى في الـEngine/السيرفر، **عمرها ما تتحط في React**.
  وفي نسخة الـSaaS بتبقى على السيرفر، والـdesktop بياخد short-lived tokens.
* لازم تنبيه واضح للمستخدم إن الطرف التاني بيسمع صوت مُولَّد. وتسجيل المكالمات ممنوع إلا بموافقة صريحة.

## 12. Folder Structure

```
AI Live Translator/
├── backend/                 # Engine (Python) — local service, later also the SaaS API
│   ├── app/
│   │   ├── core/            # config, logging, metrics/tracing
│   │   ├── services/
│   │   │   ├── audio/       # capture, playback, resample, devices, queues
│   │   │   ├── vad/         # VoiceActivityDetector interface + Silero
│   │   │   ├── stt/         # SpeechToTextProvider interface + registry
│   │   │   ├── translation/ # TranslationProvider interface + registry
│   │   │   └── tts/         # TextToSpeechProvider interface + registry
│   │   ├── providers/       # concrete implementations (faster_whisper, nllb, piper, cloud...)
│   │   ├── pipeline/        # direction pipeline (outgoing/incoming), segmenter, session
│   │   ├── api/             # REST (devices, languages, settings)
│   │   ├── websocket/       # control + transcript events channel
│   │   ├── models/          # pydantic schemas (+ DB models later)
│   │   └── utils/
│   ├── scripts/             # phase-by-phase runnable CLIs
│   └── tests/
├── native/                  # (Phase 8) process-loopback capture helper (Rust/C++)
├── desktop/                 # (later) Tauri + React UI
├── infrastructure/          # (SaaS) docker, deploy
└── docs/
```

## 13. Database (later)
بنبدأ من غير DB. لما نوصل لـHistory/SaaS هنضيف SQLite محلي (history) + PostgreSQL على السيرفر:
`users, sessions(source/target lang, platform), messages(speaker, src/tgt text, ts), user_settings, usage(minutes per stage), subscriptions`.

## 14. Roadmap — الحالة الفعلية

| # | المرحلة | الحالة | أين في الكود |
|---|---|---|---|
| 1 | الميكروفون → نص | ✅ | `services/audio/capture.py`, `providers/stt/` |
| 2 | نص → ترجمة (+ اكتشاف اللغة مع خطة بديلة) | ✅ | `providers/translation/`, `pipeline/language.py` |
| 3 | ترجمة → صوت | ✅ | `providers/tts/piper_provider.py`, `services/audio/playback.py` |
| 4 | صوت → صوت كامل | ✅ | `pipeline/voice_translator.py` |
| 5 | Streaming (نص مباشر أثناء الكلام + نطق جملة بجملة) | ✅ | `pipeline/live.py` |
| 6 | كشف الكلام (VAD) + تقسيم الجمل | ✅ | `services/vad/silero.py`, `pipeline/segmenter.py` |
| 7 | الميكروفون الوهمي | ✅ (يحتاج تثبيت VB-CABLE) | `services/audio/virtual_mic.py` |
| 8 | التقاط صوت الاجتماع (برنامج واحد أو الكل ما عدانا) | ✅ | `native/ProcessLoopback/`, `services/audio/process_capture.py` |
| 9 | الاتجاهين معًا + المقاطعات | ✅ | `engine/session.py` |
| 10 | الوضع الشامل + فحص الإعداد | ✅ | `engine/doctor.py` |
| 11–12 | Zoom / Teams / Meet / Discord | ✅ عبر الوضع الشامل · 📄 التكامل الرسمي موثّق | `docs/PLATFORMS.md` |
| 13 | واجهة المحادثة | ✅ | `app/ui/`, `app/main.py`, `app/desktop.py` |
| 14 | اجتماعات متعددة الأشخاص | ✅ جزئيًا (انظر أدناه) | `pipeline/language.py` |
| 15 | الحسابات | 📄 تصميم جاهز (يحتاج خادمًا) | `docs/SAAS.md` |
| 16 | سجل المحادثات (محلي، نص فقط، مقفول افتراضيًا) | ✅ | `engine/history.py` |
| 17 | SaaS والاشتراكات | 📄 تصميم جاهز | `docs/SAAS.md` |
| 18 | العمل بدون إنترنت | ✅ (البرنامج كله offline من الأساس) | — |
| + | ١٠٣ لغات + تنزيل الأصوات تلقائيًا | ✅ | `core/languages.py` |
| + | مثبّت exe + تنزيل النماذج عند أول تشغيل | ✅ | `packaging/`, `engine/assets.py` |

## 15. قرارات تغيّرت أثناء التنفيذ (ولماذا)

1. **الواجهة: Edge WebView2 (pywebview) بدل Tauri.** Tauri يحتاج Rust (غير مثبت)، والمحرك كله Python أصلًا.
   WebView2 مدمج في Windows 11، فالبرنامج يبقى صغيرًا. الواجهة صفحة HTML/JS، فيمكن نقلها لـTauri لاحقًا بدون إعادة كتابة.
2. **التقاط صوت الاجتماع: أداة C# خاصة بنا** (`ProcessLoopback.exe`) بدل مكتبة جاهزة.
   الأداة تستخدم واجهة ويندوز الرسمية لالتقاط صوت برنامج واحد. المكتبة الوحيدة المتاحة كانت ملفًا جاهزًا من مطوّر غير معروف.
3. **الصدى محلول من الجذر:** في الوضع الافتراضي نلتقط «كل البرامج ما عدا برنامجنا»،
   فصوت الترجمة لا يمكن أن يُلتقط ويُترجم مرة أخرى، على أي جهاز إخراج.
4. **نماذج الذكاء الاصطناعي تُنزَّل عند أول تشغيل** بدل وضعها في المثبّت (كان سيصل لـ٥ جيجا).

## 16. اجتماعات متعددة الأشخاص (المرحلة ١٤): ما يعمل وما لا يعمل

- **يعمل:** عدة أشخاص يتكلمون **بالتناوب** بلغات مختلفة، مع اختيار «لغة الطرف الآخر = اكتشاف تلقائي».
  كل جملة تُكتشف لغتها وحدها وتُترجم للغتك. الخطة البديلة تمنع القفز بين اللغات على الكلمات القصيرة.
- **لا يعمل بدقة:** شخصان يتكلمان **في نفس اللحظة** على صوت مختلط واحد. فصل الأصوات المتداخلة يحتاج نماذج ثقيلة
  (pyannote) أو صوتًا منفصلًا لكل مشارك من المنصة (Zoom RTMS / Teams bot، انظر `PLATFORMS.md`).
  المعمارية جاهزة لذلك: `LiveDirection` واحد لكل مشارك.

**MVP = المراحل ١ → ١٠ للعربية ↔ الألمانية على ويندوز: مكتمل.**
