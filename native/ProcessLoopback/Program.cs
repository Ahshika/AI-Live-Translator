// ProcessLoopback — capture the audio of ONE process tree (or of everything EXCEPT one)
// using Windows' official process-loopback API (Windows 10 2004+ / Windows 11), and write it
// to stdout as raw PCM: 16-bit signed little-endian, interleaved stereo, at --rate Hz.
//
//   ProcessLoopback.exe --pid 1234 [--mode include|exclude] [--rate 48000]
//
//   include: only that process and its children   (e.g. Zoom.exe -> just the meeting)
//   exclude: everything except that process tree   (e.g. our own engine -> no echo loop)
//
// The translator engine starts this helper and reads stdout. When the engine dies (stdin/stdout
// closed) the helper exits. Errors go to stderr as "ERROR <hresult> <message>".
// Based on Microsoft's ApplicationLoopback sample (windows-classic-samples).

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

static class Program
{
    const string VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback";
    static readonly Guid IID_IAudioClient = new("1CB9AD4C-DBFA-4c32-B178-C2F568A703B2");
    static readonly Guid IID_IAudioCaptureClient = new("C8ADBD64-E71E-48a0-A4DE-185C395CD317");

    const uint AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000;
    const uint AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000;
    const uint AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000;
    const uint AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000;
    const uint AUDCLNT_BUFFERFLAGS_SILENT = 0x2;

    static int Main(string[] args)
    {
        int pid = -1, rate = 48000;
        bool exclude = false;
        Debug = Array.IndexOf(args, "--debug") >= 0;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--pid": pid = int.Parse(args[++i]); break;
                case "--rate": rate = int.Parse(args[++i]); break;
                case "--mode": exclude = args[++i] == "exclude"; break;
                case "--check": return Environment.OSVersion.Version.Build >= 19041 ? 0 : 3;
            }
        }
        if (pid <= 0)
        {
            Console.Error.WriteLine("usage: ProcessLoopback --pid <pid> [--mode include|exclude] [--rate 48000]");
            return 2;
        }
        try
        {
            Run(pid, exclude, rate);
            return 0;
        }
        catch (COMException e)
        {
            Console.Error.WriteLine($"ERROR 0x{e.HResult:X8} {e.Message}");
            return 1;
        }
        catch (IOException)
        {
            return 0; // engine closed the pipe: normal shutdown
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"ERROR 0x{e.HResult:X8} {e.Message}");
            return 1;
        }
    }

    static bool Debug;

    static void Run(int pid, bool exclude, int rate)
    {
        IAudioClient client = Activate(pid, exclude);

        // Process-loopback clients have no mix format: we choose one, Windows converts.
        var fmt = new WAVEFORMATEX
        {
            wFormatTag = 1, // PCM
            nChannels = 2,
            nSamplesPerSec = (uint)rate,
            wBitsPerSample = 16,
            nBlockAlign = 4,
            nAvgBytesPerSec = (uint)rate * 4,
            cbSize = 0,
        };
        IntPtr fmtPtr = Marshal.AllocHGlobal(Marshal.SizeOf<WAVEFORMATEX>());
        Marshal.StructureToPtr(fmt, fmtPtr, false);
        const long bufferDuration = 2_000_000; // 200 ms in 100-ns units
        client.Initialize(0 /*shared*/, AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK
                          | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                          bufferDuration, 0, fmtPtr, IntPtr.Zero);

        using var ready = new AutoResetEvent(false);
        client.SetEventHandle(ready.SafeWaitHandle.DangerousGetHandle());
        Guid iidCapture = IID_IAudioCaptureClient;
        client.GetService(ref iidCapture, out object captureObj);
        var capture = (IAudioCaptureClient)captureObj;

        using Stream stdout = Console.OpenStandardOutput();
        byte[] buffer = new byte[rate * 4]; // 1 s worth, reused
        client.Start();
        Console.Error.WriteLine($"READY {rate} 2 s16le");
        long packets = 0, silentPackets = 0, totalFrames = 0, wakeups = 0, signalled = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            while (true)
            {
                if (ready.WaitOne(100)) signalled++;
                wakeups++;
                if (Debug && sw.ElapsedMilliseconds > 500)
                {
                    Console.Error.WriteLine($"DEBUG wake={wakeups} sig={signalled} packets={packets} silent={silentPackets} frames={totalFrames}");
                    sw.Restart();
                }
                while (true)
                {
                    capture.GetNextPacketSize(out uint packet);
                    if (packet == 0) break;
                    capture.GetBuffer(out IntPtr data, out uint frames, out uint flags, out _, out _);
                    int bytes = (int)frames * 4;
                    packets++; totalFrames += frames;
                    if ((flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0) silentPackets++;
                    if (bytes > buffer.Length) buffer = new byte[bytes];
                    if ((flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0) Array.Clear(buffer, 0, bytes);
                    else Marshal.Copy(data, buffer, 0, bytes);
                    capture.ReleaseBuffer(frames);
                    stdout.Write(buffer, 0, bytes);
                }
                stdout.Flush();
            }
        }
        finally
        {
            client.Stop();
            Marshal.FreeHGlobal(fmtPtr);
        }
    }

    static IAudioClient Activate(int pid, bool exclude)
    {
        var p = new AUDIOCLIENT_ACTIVATION_PARAMS
        {
            ActivationType = 1, // AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK
            TargetProcessId = (uint)pid,
            ProcessLoopbackMode = exclude ? 1 : 0, // EXCLUDE_TARGET_PROCESS_TREE : INCLUDE_TARGET_PROCESS_TREE
        };
        int size = Marshal.SizeOf<AUDIOCLIENT_ACTIVATION_PARAMS>();
        IntPtr blob = Marshal.AllocHGlobal(size);
        Marshal.StructureToPtr(p, blob, false);
        var pv = new PROPVARIANT { vt = 65 /*VT_BLOB*/, cbSize = (uint)size, pBlobData = blob };
        IntPtr pvPtr = Marshal.AllocHGlobal(Marshal.SizeOf<PROPVARIANT>());
        Marshal.StructureToPtr(pv, pvPtr, false);
        try
        {
            var handler = new CompletionHandler();
            ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, IID_IAudioClient, pvPtr, handler, out _);
            if (!handler.Done.WaitOne(5000)) throw new TimeoutException("Audio activation timed out");
            Marshal.ThrowExceptionForHR(handler.Result);
            return (IAudioClient)handler.Client!;
        }
        finally
        {
            Marshal.FreeHGlobal(pvPtr);
            Marshal.FreeHGlobal(blob);
        }
    }

    // ---- interop -------------------------------------------------------------------------

    [DllImport("Mmdevapi.dll", ExactSpelling = true, PreserveSig = false)]
    static extern void ActivateAudioInterfaceAsync(
        [MarshalAs(UnmanagedType.LPWStr)] string deviceInterfacePath,
        [MarshalAs(UnmanagedType.LPStruct)] Guid riid,
        IntPtr activationParams,
        IActivateAudioInterfaceCompletionHandler completionHandler,
        out IActivateAudioInterfaceAsyncOperation activationOperation);

    [StructLayout(LayoutKind.Sequential)]
    struct AUDIOCLIENT_ACTIVATION_PARAMS
    {
        public int ActivationType;
        public uint TargetProcessId;
        public int ProcessLoopbackMode;
    }

    [StructLayout(LayoutKind.Explicit, Size = 24)]
    struct PROPVARIANT
    {
        [FieldOffset(0)] public ushort vt;
        [FieldOffset(8)] public uint cbSize;
        [FieldOffset(16)] public IntPtr pBlobData;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 2)]
    struct WAVEFORMATEX
    {
        public ushort wFormatTag;
        public ushort nChannels;
        public uint nSamplesPerSec;
        public uint nAvgBytesPerSec;
        public ushort nBlockAlign;
        public ushort wBitsPerSample;
        public ushort cbSize;
    }

    [ComImport, Guid("72A22D78-CDE4-431D-B8CC-843A71199B6D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IActivateAudioInterfaceAsyncOperation
    {
        void GetActivateResult(out int activateResult, [MarshalAs(UnmanagedType.IUnknown)] out object activatedInterface);
    }

    [ComImport, Guid("41D949AB-9862-444A-80F6-C261334DA5EB"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IActivateAudioInterfaceCompletionHandler
    {
        void ActivateCompleted(IActivateAudioInterfaceAsyncOperation activateOperation);
    }

    // Marker: the callback may arrive on any thread.
    [ComImport, Guid("94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IAgileObject { }

    [ComVisible(true)]
    sealed class CompletionHandler : IActivateAudioInterfaceCompletionHandler, IAgileObject
    {
        public readonly ManualResetEvent Done = new(false);
        public int Result = unchecked((int)0x80004005);
        public object? Client;

        public void ActivateCompleted(IActivateAudioInterfaceAsyncOperation op)
        {
            try
            {
                op.GetActivateResult(out int hr, out object iface);
                Result = hr;
                Client = iface;
            }
            catch (Exception e)
            {
                Result = e.HResult;
            }
            finally
            {
                Done.Set();
            }
        }
    }

    [ComImport, Guid("1CB9AD4C-DBFA-4c32-B178-C2F568A703B2"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IAudioClient
    {
        void Initialize(int shareMode, uint streamFlags, long hnsBufferDuration, long hnsPeriodicity,
                        IntPtr pFormat, IntPtr audioSessionGuid);
        void GetBufferSize(out uint bufferFrames);
        void GetStreamLatency(out long latency);
        void GetCurrentPadding(out uint padding);
        [PreserveSig] int IsFormatSupported(int shareMode, IntPtr pFormat, out IntPtr closestMatch);
        void GetMixFormat(out IntPtr deviceFormat);
        void GetDevicePeriod(out long defaultPeriod, out long minimumPeriod);
        void Start();
        void Stop();
        void Reset();
        void SetEventHandle(IntPtr eventHandle);
        void GetService(ref Guid riid, [MarshalAs(UnmanagedType.IUnknown)] out object service);
    }

    [ComImport, Guid("C8ADBD64-E71E-48a0-A4DE-185C395CD317"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IAudioCaptureClient
    {
        void GetBuffer(out IntPtr data, out uint numFramesToRead, out uint flags,
                       out ulong devicePosition, out ulong qpcPosition);
        void ReleaseBuffer(uint numFramesRead);
        void GetNextPacketSize(out uint numFramesInNextPacket);
    }
}
