# دعم منصات الاجتماعات (المرحلتان ١١ و١٢)

> تم التحقق من هذه المعلومات من المصادر الرسمية في **أكتوبر ٢٠٢٦**. المنصات تغيّر سياساتها، فراجعها قبل أي تنفيذ.

## الخلاصة

**الوضع الشامل** (الميكروفون الوهمي + التقاط صوت البرنامج) هو الحل الأساسي، وهو **منفّذ ويعمل** مع كل المنصات بدون أي حساب مطوّر أو موافقة:

| المنصة | الوضع الشامل (يعمل الآن) | تكامل رسمي ممكن لاحقًا | يستطيع **إرسال** صوت مترجم؟ |
|---|---|---|---|
| Zoom | ✅ اختر «CABLE Output» كميكروفون، والتقاط Zoom.exe | RTMS أو Meeting SDK | RTMS: **لا**. Meeting SDK: نعم (كبوت مشارك) |
| Microsoft Teams | ✅ نفس الطريقة | بوت وسائط مستضاف (Real-time Media Platform) | نعم، لكن عبر بوت على خادم Windows Server |
| Google Meet | ✅ عبر المتصفح (التقاط chrome/msedge) | Meet Media API | لا (استقبال فقط)، و**لا يقبل تسجيلات جديدة** |
| Discord | ✅ نفس الطريقة | لا يوجد مسار رسمي لاستقبال الصوت | — |
| واتساب / تيليجرام / سكايب / أي برنامج | ✅ | — | — |

## التفاصيل

### Zoom
- **RTMS (Realtime Media Streams):**
  - يرسل صوت كل مشارك **منفصلًا**، بصيغة PCM 16kHz أحادي، ومع كل حزمة اسم المتحدث، عبر WebSocket إلى خادمك.
  - ممتاز للاجتماعات الجماعية، لأنه يغني عن فصل المتحدثين.
  - **لكنه للاستقبال فقط** ولا يسمح بإرسال صوت للاجتماع، فلا يغني عن الميكروفون الوهمي.
  - يحتاج تطبيق Zoom مسجّلًا وموافقات وخادمًا عامًا يستقبل الاتصالات، أي أنه غير مناسب لميزانية صفر حاليًا.
- **Meeting SDK (Raw Data):**
  - يسمح ببوت يدخل الاجتماع كمشارك، ويستقبل الصوت الخام لكل مستخدم، و**يرسل صوتًا مخصصًا كأنه ميكروفون**، بحد أقصى ١٠٠ ميلي ثانية لكل استدعاء إرسال.
  - العيب أن الترجمة تظهر كمشارك إضافي باسم البوت، لا بصوتك أنت.
- **القرار:** الوضع الشامل للمستخدم العادي. وRTMS خيار مستقبلي لنسخة الشركات، لأنه يعطي صوت كل متحدث منفصلًا.

### Microsoft Teams
- البوتات ذات «الوسائط المستضافة في التطبيق» ترسل وتستقبل الصوت إطارًا بإطار: ٢٠ ميلي ثانية، و١٦kHz، و١٦ بت، أي نفس صيغتنا الداخلية بالضبط. وتعرف من المتحدث في كل إطار.
- **المتطلبات:** مكتبة ‎.NET (Microsoft.Graph.Communications.Calls.Media)، وتشغيل البوت على **Windows Server** (أو Azure)، وتسجيل تطبيق في Azure.
- **القرار:** مناسب لنسخة الشركات المدفوعة فقط. الوضع الشامل يغطي الأفراد الآن.

### Google Meet
- Meet Media API يعطي صوتًا لحظيًا من الاجتماع، لكنه ضمن برنامج المعاينة للمطورين (Developer Preview).
- يجب أن يكون **كل المشاركين** مسجلين في ذلك البرنامج.
- **لم يعد يقبل تسجيلات جديدة.**
- **القرار:** غير عملي. نستخدم الوضع الشامل: Meet يعمل داخل المتصفح، فنلتقط صوت chrome.exe أو msedge.exe.

### Discord
- Discord **لا يدعم رسميًا** استقبال البوتات للصوت. المكتبات التي تفعل ذلك تعتمد على هندسة عكسية، وقد تتعطل في أي وقت.
- أتمتة حساب المستخدم (selfbot) مخالفة لشروط Discord، ولن نفعلها.
- **القرار:** الوضع الشامل فقط.

## كيف تُضاف منصة رسمية لاحقًا (بدون تغيير باقي البرنامج)

المحرك لا يعرف من أين يأتي الصوت. كل ما يحتاجه هو «مصدر صوت» (`AudioSource`) و«مخرج صوت» (`AudioSink`):

```
RTMS / Teams bot ──► AudioSource لكل مشارك ──► LiveDirection (مترجم واحد لكل متحدث) ──► السماعات
Meeting SDK bot  ◄── AudioSink (بدل الميكروفون الوهمي) ◄── ترجمة كلامي
```

أي أن تكامل Zoom RTMS = كلاس يحوّل حزم WebSocket إلى `read_frame()`، ويشغّل `LiveDirection` لكل `user_id`.

## المصادر
- [Zoom RTMS — Handling media data](https://developers.zoom.us/docs/rtms/meetings/media/)
- [Zoom RTMS — Event reference](https://developers.zoom.us/docs/rtms/event-reference/)
- [Zoom Meeting SDK — Raw data](https://developers.zoom.us/docs/meeting-sdk/ios/add-features/raw-data/)
- [Teams — Real-time media calls and meetings](https://learn.microsoft.com/en-us/microsoftteams/platform/bots/calls-and-meetings/real-time-media-concepts)
- [Google Meet Media API overview](https://developers.google.com/workspace/meet/media-api/guides/overview)
- [Discord voice docs](https://discord.com/developers/docs/resources/voice) · [discord-api-docs #808](https://github.com/discord/discord-api-docs/issues/808)
