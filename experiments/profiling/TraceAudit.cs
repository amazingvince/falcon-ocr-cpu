// Offline only. No recorder, symbol download, process launch, or workload execution.
// API: PerfView 3.2.6 / TraceEvent 3.2.6+9a99c8202310fc0ab2f84690881f6a4a4ada0c44.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Web.Script.Serialization;
using Microsoft.Diagnostics.Tracing;
using Microsoft.Diagnostics.Tracing.Etlx;
using Microsoft.Diagnostics.Tracing.Parsers.Kernel;

internal static class TraceAudit
{
    internal const string TraceEventHash = "530946dc20e89754783f0ac76d86b7a4eedf95326f255db34c32fcbf5c3ce0ff";
    internal static readonly Dictionary<string, string> Dependencies = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    internal static string DependencyDirectory;
    internal static Dictionary<string, object> Obj(params object[] pairs)
    {
        var d = new Dictionary<string, object>();
        for (int i = 0; i < pairs.Length; i += 2) d.Add((string)pairs[i], pairs[i + 1]);
        return d;
    }
    internal static string Hash(string path)
    {
        using (var f = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (var h = SHA256.Create()) return BitConverter.ToString(h.ComputeHash(f)).Replace("-", "").ToLowerInvariant();
    }
    internal static void Need(bool ok, string why) { if (!ok) throw new InvalidDataException(why); }
    internal static object Utc(DateTime date)
    {
        if (date == DateTime.MinValue || date == DateTime.MaxValue || date.Kind == DateTimeKind.Unspecified) return null;
        return date.ToUniversalTime().ToString("O", CultureInfo.InvariantCulture);
    }
    internal static Dictionary<string, string> Parse(string[] args)
    {
        var result = new Dictionary<string, string>();
        var allowed = new HashSet<string>(new[] { "--etl", "--pid", "--output", "--bin-ms", "--expected-image", "--expected-command-line" });
        Need(args.Length % 2 == 0, "Arguments must be --name value pairs");
        for (int i = 0; i < args.Length; i += 2)
        {
            Need(allowed.Contains(args[i]) && !result.ContainsKey(args[i]), "Unknown or repeated argument: " + args[i]);
            result.Add(args[i], args[i + 1]);
        }
        foreach (string k in new[] { "--etl", "--pid", "--output" }) Need(result.ContainsKey(k), "Required argument: " + k);
        return result;
    }
    public static int Main(string[] args)
    {
        if (args.Length == 1 && args[0] == "--self-test") return SelfTest();
        try
        {
            var options = Parse(args);
            DependencyDirectory = Environment.GetEnvironmentVariable("TRACEAUDIT_DEPENDENCIES");
            Need(!String.IsNullOrEmpty(DependencyDirectory) && Directory.Exists(DependencyDirectory), "Set TRACEAUDIT_DEPENDENCIES to the extracted PerfView 3.2.6 directory");
            DependencyDirectory = Path.GetFullPath(DependencyDirectory);
            foreach (string p in Directory.GetFiles(DependencyDirectory, "*.dll")) Dependencies.Add(Path.GetFullPath(p), Hash(p));
            Need(Dependencies[Path.Combine(DependencyDirectory, "Microsoft.Diagnostics.Tracing.TraceEvent.dll")] == TraceEventHash, "Wrong TraceEvent assembly bytes");
            AppDomain.CurrentDomain.AssemblyResolve += delegate(object sender, ResolveEventArgs request)
            {
                string p = Path.Combine(DependencyDirectory, new AssemblyName(request.Name).Name + ".dll");
                if (!Dependencies.ContainsKey(p)) return null;
                Need(Hash(p) == Dependencies[p], "Dependency changed before load: " + p);
                return Assembly.LoadFrom(p);
            };
            return Run(options);
        }
        catch (Exception e) { Console.Error.WriteLine(e.ToString()); return 2; }
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int Run(Dictionary<string, string> args)
    {
        string input = Path.GetFullPath(args["--etl"]), output = Path.GetFullPath(args["--output"]);
        int pid; double binMs;
        Need(Int32.TryParse(args["--pid"], out pid) && pid > 0, "PID must be positive");
        Need(Double.TryParse(args.ContainsKey("--bin-ms") ? args["--bin-ms"] : "1000", NumberStyles.Float, CultureInfo.InvariantCulture, out binMs) && !Double.IsNaN(binMs) && !Double.IsInfinity(binMs) && binMs > 0, "Invalid bin width");
        Need(File.Exists(input) && String.Equals(Path.GetExtension(input), ".etl", StringComparison.OrdinalIgnoreCase), "Input must be a completed merged .etl file; existing ETLX is not accepted");
        string converted = output + ".etlx", conversionLog = output + ".conversion.log";
        Need(!File.Exists(output) && !File.Exists(converted) && !File.Exists(conversionLog), "Output, ETLX and conversion log must all be fresh");
        Need(Directory.Exists(Path.GetDirectoryName(output)), "Output parent must already exist");
        var report = Obj("schema_version", 1, "kind", "offline-exact-pid-etl-audit-v1", "status", "started", "target_pid", pid,
            "started_utc", DateTime.UtcNow.ToString("O"), "arguments", args, "failures", new List<string>(),
            "symbol_resolution_performed", false, "new_etlx_conversion", true,
            "profile_quality_policy", Obj("minimum_attached_stack_fraction", 0.99, "require_real_start_and_stop", true, "require_same_name_unique", true,
                "require_zero_available_loss_counters", true, "separate_buffer_loss_count_available", false),
            "limitations", new[] { "ETW EventsLost is the recorded file/source count, not a proof that every possible provider emitted every event.", "Missing stacks remain missing; stack presence does not establish unwind or symbol completeness.", "TraceProcess lifetime can be clipped to the trace; only separate ProcessStart/ProcessStop records establish observed boundaries.", "Module/PDB fields come from this trace, not current disk files or symbol servers. Empty/default fields remain explicitly unavailable.", "CPU samples are observations; Count payload totals are reported separately and are not elapsed wall time." });
        var failures = (List<string>)report["failures"];
        string initialHash = null;
        int result = 2;
        using (var destination = new FileStream(output, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
        {
            try
            {
                // Keep the original open with no write/delete sharing through conversion and audit.
                using (var inputLock = new FileStream(input, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    initialHash = Hash(input);
                    report["input"] = Obj("path", input, "bytes", inputLock.Length, "sha256", initialHash);
                    report["helper"] = Obj("path", Assembly.GetExecutingAssembly().Location, "sha256", Hash(Assembly.GetExecutingAssembly().Location));
                    var assembly = typeof(TraceLog).Assembly;
                    Need(assembly.GetName().Version.ToString() == "3.2.6.0" && Hash(assembly.Location) == TraceEventHash, "Loaded TraceEvent identity differs");
                    report["trace_event"] = Obj("path", assembly.Location, "sha256", Hash(assembly.Location), "assembly", assembly.FullName,
                        "informational_version", assembly.GetCustomAttributes(typeof(AssemblyInformationalVersionAttribute), false).Cast<AssemblyInformationalVersionAttribute>().Single().InformationalVersion,
                        "api_source", "https://github.com/microsoft/perfview/blob/9a99c8202310fc0ab2f84690881f6a4a4ada0c44/src/TraceEvent/TraceLog.cs");
                    var lifecycle = new List<object>();
                    var lossCallbacks = new List<object>();
                    int rawLostBefore, rawLostAfter;
                    using (var logWriter = new StreamWriter(new FileStream(conversionLog, FileMode.CreateNew, FileAccess.Write, FileShare.Read), new UTF8Encoding(false)))
                    using (var raw = new ETWTraceEventSource(input))
                    {
                        Action<string, ProcessTraceData> observe = delegate(string kind, ProcessTraceData e)
                        {
                            if (e.ProcessID != pid) return;
                            lifecycle.Add(Obj("kind", kind, "timestamp_utc", Utc(e.TimeStamp), "relative_ms", e.TimeStampRelativeMSec,
                                "pid", e.ProcessID, "parent_pid", e.ParentID, "image", e.ImageFileName, "command_line", e.CommandLine,
                                "unique_process_key_hex", "0x" + e.UniqueProcessKey.ToString("x"), "exit_status", kind == "stop" ? (object)e.ExitStatus : null));
                        };
                        raw.Kernel.ProcessStart += delegate(ProcessTraceData e) { observe("start", e); };
                        raw.Kernel.ProcessStop += delegate(ProcessTraceData e) { observe("stop", e); };
                        raw.Kernel.ProcessDCStart += delegate(ProcessTraceData e) { observe("rundown_start", e); };
                        raw.Kernel.ProcessDCStop += delegate(ProcessTraceData e) { observe("rundown_stop", e); };
                        rawLostBefore = raw.EventsLost;
                        report["raw_session"] = Obj("name", raw.SessionName, "start_utc", Utc(raw.SessionStartTime), "end_utc", Utc(raw.SessionEndTime),
                            "start_datetime_kind", raw.SessionStartTime.Kind.ToString(), "end_datetime_kind", raw.SessionEndTime.Kind.ToString(),
                            "duration_ms", raw.SessionEndTimeRelativeMSec, "timer_resolution_100ns", raw.TimerResolution);
                        var options = new TraceLogOptions { KeepAllEvents = true, ContinueOnError = false, MaxEventCount = Int32.MaxValue,
                            SkipMSec = 0, LocalSymbolsOnly = true, AlwaysResolveSymbols = false, ShouldResolveSymbols = delegate(string p) { return false; }, ConversionLog = logWriter };
                        options.OnLostEvents = delegate(bool truncated, int lost, int events) { lossCallbacks.Add(Obj("truncated", truncated, "events_lost", lost, "event_count", events)); };
                        TraceLog.CreateFromEventTraceLogFile(raw, converted, options);
                        rawLostAfter = raw.EventsLost;
                    }
                    report["target_lifecycle_events"] = lifecycle;
                    report["loss_callbacks"] = lossCallbacks;
                    if (lossCallbacks.Cast<Dictionary<string, object>>().Any(c => LossCallbackFails((bool)c["truncated"], (int)c["events_lost"])))
                        failures.Add("Conversion callback reported lost events or truncation");
                    using (var log = new TraceLog(converted))
                    {
                        report["session"] = Obj("start_utc", Utc(log.SessionStartTime), "end_utc", Utc(log.SessionEndTime), "duration_ms", log.SessionEndTimeRelativeMSec,
                            "collection_utc_offset_minutes", log.UTCOffsetMinutes, "machine_name", log.MachineName, "os_build", log.OSBuild,
                            "processor_count", log.NumberOfProcessors, "sample_profile_interval_ms_first", log.SampleProfileInterval.TotalMilliseconds,
                            "has_any_stacks", log.HasCallStacks, "has_pdb_info", log.HasPdbInfo, "event_count", log.EventCount,
                            "first_time_inversion_event_index", log.FirstTimeInversion == EventIndex.Invalid ? null : (object)(uint)log.FirstTimeInversion);
                        report["loss"] = Obj("raw_events_lost_before", rawLostBefore, "raw_events_lost_after", rawLostAfter, "etlx_events_lost", log.EventsLost,
                            "etlx_truncated", log.Truncated, "counts_agree", rawLostBefore == rawLostAfter && rawLostAfter == log.EventsLost,
                            "recorded_zero_available_event_loss", rawLostBefore == 0 && rawLostAfter == 0 && log.EventsLost == 0 && !log.Truncated && !lossCallbacks.Cast<Dictionary<string, object>>().Any(c => LossCallbackFails((bool)c["truncated"], (int)c["events_lost"])),
                            "separate_buffers_lost", null, "separate_buffers_lost_status", "unknown: not exposed by the used public TraceEvent API; inspect recorder evidence separately");
                        if (rawLostBefore != 0 || rawLostAfter != 0 || log.EventsLost != 0 || log.Truncated) failures.Add("Recorded lost events or truncated conversion");
                        if (rawLostBefore != rawLostAfter || rawLostAfter != log.EventsLost) failures.Add("Raw and ETLX loss counts disagree");
                        if (Utc(log.SessionStartTime) == null || Utc(log.SessionEndTime) == null) failures.Add("Session UTC is unavailable");
                        var matches = log.Processes.Where(p => p.ProcessID == pid).ToList();
                        report["pid_matches"] = matches.Select(p => (object)ProcessInfo(p)).ToList();
                        Need(matches.Count == 1, "Exact target PID has " + matches.Count + " process lifetimes; missing or reused/ambiguous PID");
                        var target = matches[0];
                        report["target"] = ProcessInfo(target);
                        var sameName = log.Processes.Where(p => String.Equals(p.Name, target.Name, StringComparison.OrdinalIgnoreCase)).ToList();
                        report["same_name_processes"] = sameName.Select(p => (object)ProcessInfo(p)).ToList();
                        report["same_name_unique"] = sameName.Count == 1;
                        if (String.IsNullOrEmpty(target.ImageFileName) || String.IsNullOrEmpty(target.CommandLine)) failures.Add("Target image or command line unavailable");
                        if (args.ContainsKey("--expected-image") && !String.Equals(Path.GetFileName(target.ImageFileName), args["--expected-image"], StringComparison.OrdinalIgnoreCase)) failures.Add("Target image basename differs");
                        if (args.ContainsKey("--expected-command-line") && !String.Equals(target.CommandLine, args["--expected-command-line"], StringComparison.Ordinal)) failures.Add("Target command line differs");
                        var lifecycleRows = lifecycle.Cast<Dictionary<string, object>>().ToList();
                        int starts = lifecycleRows.Count(p => (string)p["kind"] == "start"), stops = lifecycleRows.Count(p => (string)p["kind"] == "stop");
                        report["lifetime_observation"] = Obj("start_events", starts, "stop_events", stops, "both_boundaries_observed_once", starts == 1 && stops == 1,
                            "trace_process_times_may_be_clipped", starts != 1 || stops != 1);
                        var modules = new List<object>();
                        foreach (var loaded in target.LoadedModules)
                        {
                            var m = loaded.ModuleFile;
                            modules.Add(Obj("path", loaded.FilePath, "load_base_hex", "0x" + loaded.ImageBase.ToString("x"), "load_relative_ms", loaded.LoadTimeRelativeMSec,
                                "unload_relative_ms", loaded.UnloadTimeRelativeMSec, "module_file_index", (int)m.ModuleFileIndex,
                                "image_size", m.ImageSize, "image_checksum", m.ImageChecksum, "image_id", m.ImageId,
                                "file_version", EmptyNull(m.FileVersion), "product_version", EmptyNull(m.ProductVersion), "build_time_utc", Utc(m.BuildTime),
                                "pdb_name", EmptyNull(m.PdbName), "pdb_signature", m.PdbSignature == Guid.Empty ? null : m.PdbSignature.ToString("D"),
                                "pdb_age", m.PdbSignature == Guid.Empty ? null : (object)m.PdbAge,
                                "pdb_identity_available", m.PdbSignature != Guid.Empty && !String.IsNullOrEmpty(m.PdbName), "binary_format", m.BinaryFormat.ToString()));
                        }
                        report["modules"] = modules;
                        var bins = new SortedDictionary<long, Bin>();
                        var threads = new SortedDictionary<int, long>();
                        long total = 0, withStack = 0, payloadCount = 0, wrongOwner = 0, outOfBounds = 0, dpc = 0, isr = 0, nonProcess = 0;
                        double? first = null, last = null;
                        foreach (var ev in log.Events)
                        {
                            var sample = ev as SampledProfileTraceData;
                            if (sample == null || sample.ProcessID != pid) continue;
                            var owner = sample.Process();
                            if (owner == null || owner.ProcessIndex != target.ProcessIndex) { wrongOwner++; continue; }
                            double time = sample.TimeStampRelativeMSec;
                            if (Double.IsNaN(time) || Double.IsInfinity(time) || time < 0 || time > log.SessionEndTimeRelativeMSec || time < target.StartTimeRelativeMsec || time > target.EndTimeRelativeMsec) { outOfBounds++; continue; }
                            bool stack = log.GetCallStackIndexForEvent(sample) != CallStackIndex.Invalid;
                            total++; if (stack) withStack++; payloadCount += sample.Count;
                            if (sample.ExecutingDPC) dpc++; if (sample.ExecutingISR) isr++; if (sample.NonProcess) nonProcess++;
                            first = first.HasValue ? Math.Min(first.Value, time) : time; last = last.HasValue ? Math.Max(last.Value, time) : time;
                            long slot = checked((long)Math.Floor(time / binMs));
                            Bin bin; if (!bins.TryGetValue(slot, out bin)) { bin = new Bin(); bins.Add(slot, bin); }
                            bin.Count++; if (stack) bin.WithStack++; bin.PayloadCount += sample.Count;
                            long threadCount; threads.TryGetValue(sample.ThreadID, out threadCount); threads[sample.ThreadID] = threadCount + 1;
                        }
                        report["samples"] = Obj("event_count", total, "with_attached_stack", withStack, "without_attached_stack", total - withStack,
                            "attached_stack_fraction", total == 0 ? null : (object)((double)withStack / total), "minimum_required_attached_stack_fraction", 0.99,
                            "count_payload_total", payloadCount, "first_relative_ms", first, "last_relative_ms", last,
                            "first_utc", first.HasValue ? Utc(log.SessionStartTime.AddMilliseconds(first.Value)) : null,
                            "last_utc", last.HasValue ? Utc(log.SessionStartTime.AddMilliseconds(last.Value)) : null,
                            "wrong_or_missing_process_index_count", wrongOwner, "out_of_bounds_count", outOfBounds,
                            "dpc_count", dpc, "isr_count", isr, "non_process_count", nonProcess,
                            "thread_counts", threads.Select(p => Obj("thread_id", p.Key, "samples", p.Value)).ToList());
                        report["sample_bins"] = Obj("width_ms", binMs, "origin", "session_start", "interval", "[start,end)", "omitted_bins_have_zero_samples", true,
                            "nonempty", bins.Select(p => Obj("index", p.Key, "start_relative_ms", p.Key * binMs, "end_relative_ms", (p.Key + 1) * binMs,
                                "samples", p.Value.Count, "with_attached_stack", p.Value.WithStack, "without_attached_stack", p.Value.Count - p.Value.WithStack, "count_payload_total", p.Value.PayloadCount)).ToList());
                        if (total == 0) failures.Add("No target sampled-profile events");
                        if (wrongOwner != 0 || outOfBounds != 0) failures.Add("Samples have ambiguous ownership or lie outside target/session bounds");
                        failures.AddRange(CoverageFailures(starts, stops, sameName.Count, total, withStack));
                    }
                    Need(Hash(input) == initialHash, "Input ETL changed during audit");
                    report["input_unchanged"] = true;
                    report["converted_etlx"] = Obj("path", converted, "sha256", Hash(converted), "bytes", new FileInfo(converted).Length);
                    report["conversion_log"] = Obj("path", conversionLog, "sha256", Hash(conversionLog));
                }
                result = failures.Count == 0 ? 0 : 1;
                report["status"] = result == 0 ? "audit_passed" : "audit_failed";
            }
            catch (Exception e) { failures.Add(e.Message); report["status"] = "audit_error"; report["error"] = e.ToString(); result = 2; }
            finally
            {
                var loaded = new List<object>();
                foreach (var a in AppDomain.CurrentDomain.GetAssemblies())
                {
                    if (a.IsDynamic || String.IsNullOrEmpty(a.Location) || !Dependencies.ContainsKey(a.Location)) continue;
                    string after = Hash(a.Location); bool unchanged = after == Dependencies[a.Location];
                    loaded.Add(Obj("name", a.FullName, "path", a.Location, "sha256_before", Dependencies[a.Location], "sha256_after", after, "unchanged", unchanged));
                    if (!unchanged) { failures.Add("Loaded dependency changed: " + a.Location); report["status"] = "audit_error"; result = 2; }
                }
                report["loaded_dependencies"] = loaded;
                report["ended_utc"] = DateTime.UtcNow.ToString("O");
                report["exit_code"] = result;
                var serializer = new JavaScriptSerializer { MaxJsonLength = Int32.MaxValue, RecursionLimit = 100 };
                byte[] data = new UTF8Encoding(false).GetBytes(serializer.Serialize(report) + "\n");
                destination.Write(data, 0, data.Length); destination.Flush(true);
            }
        }
        Console.WriteLine("{0}: {1}", report["status"], output);
        return result;
    }
    private static object EmptyNull(string value) { return String.IsNullOrEmpty(value) ? null : value; }
    private static Dictionary<string, object> ProcessInfo(TraceProcess p)
    {
        return Obj("pid", p.ProcessID, "process_index", (int)p.ProcessIndex, "name", p.Name, "image", EmptyNull(p.ImageFileName),
            "command_line", EmptyNull(p.CommandLine), "parent_pid", p.ParentID, "start_utc", Utc(p.StartTime), "end_utc", Utc(p.EndTime),
            "start_relative_ms", p.StartTimeRelativeMsec, "end_relative_ms", p.EndTimeRelativeMsec, "exit_status", p.ExitStatus,
            "is_64_bit", p.Is64Bit, "sample_based_cpu_ms", p.CPUMSec);
    }
    private sealed class Bin { public long Count, WithStack, PayloadCount; }
    private static bool LossCallbackFails(bool truncated, int lost) { return truncated || lost != 0; }
    private static List<string> CoverageFailures(int starts, int stops, int sameNames, long samples, long stacked)
    {
        var failures = new List<string>();
        if (starts != 1 || stops != 1) failures.Add("Exactly one actual process start and stop are required for complete-lifetime coverage");
        if (sameNames != 1) failures.Add("Same-name process ambiguity prevents the separate name-based PerfView stack export join");
        if (samples <= 0 || stacked < 0 || stacked > samples || stacked * 100 < samples * 99) failures.Add("Attached-stack coverage is below the prospectively fixed 99 percent threshold or has invalid counts");
        return failures;
    }
    private static int SelfTest()
    {
        int passed = 0;
        Need(Parse(new[] { "--etl", "a.etl", "--pid", "123", "--output", "a.json" })["--pid"] == "123", "parse"); passed++;
        foreach (string[] input in new[] { new[] { "--etl" }, new[] { "--bad", "x" }, new[] { "--etl", "a", "--etl", "b" } })
        { bool rejected = false; try { Parse(input); } catch (InvalidDataException) { rejected = true; } Need(rejected, "malformed arguments accepted"); passed++; }
        Need(Utc(DateTime.SpecifyKind(new DateTime(2026, 1, 1), DateTimeKind.Unspecified)) == null, "unspecified UTC invented"); passed++;
        Need((string)Utc(new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc)) == "2026-01-01T00:00:00.0000000Z", "UTC serialization"); passed++;
        Need(CoverageFailures(1, 1, 1, 20, 20).Count == 0, "Complete sample coverage rejected"); passed++;
        foreach (int[] c in new[] { new[] { 0, 1, 1, 20, 20 }, new[] { 1, 0, 1, 20, 20 }, new[] { 2, 1, 1, 20, 20 },
            new[] { 1, 1, 2, 20, 20 }, new[] { 1, 1, 1, 0, 0 }, new[] { 1, 1, 1, 20, 0 }, new[] { 1, 1, 1, 20, 19 } })
        { Need(CoverageFailures(c[0], c[1], c[2], c[3], c[4]).Count > 0, "Partial profile falsely accepted"); passed++; }
        Need(CoverageFailures(1, 1, 1, 100, 99).Count == 0, "Exact 99 percent threshold rejected"); passed++;
        Need(CoverageFailures(1, 1, 1, 100, 98).Count > 0, "98 percent coverage accepted"); passed++;
        Need(CoverageFailures(1, 1, 1, 100, 101).Count > 0, "Invalid stack overcount accepted"); passed++;
        Need(LossCallbackFails(true, 0), "Truncated callback accepted"); passed++;
        Need(LossCallbackFails(false, 1), "Lost-event callback accepted"); passed++;
        Need(!LossCallbackFails(false, 0), "Zero-loss callback rejected"); passed++;
        Console.WriteLine("TraceAudit host self-tests passed: " + passed); return 0;
    }
}
