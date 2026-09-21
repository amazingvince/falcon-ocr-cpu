// Offline retained-ETLX diagnostic. No recording, conversion, symbols or model.
// Pinned TraceEvent 3.2.6 API, shared with TraceAudit / inspect_trace_boundary.
using System;
using System.Collections.Generic;
using System.Diagnostics;
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

internal static class CompactIpHistogram
{
    const string AuditSha = "37d46af7873aec974f663d6dc8187b3df1322511b58ef8e7b671f23b681f20b5";
    const string ReceiptSha = "ea4842b0289c971a72f6e22bb0d5d10b623743376685159e34ef894444f3acf3";
    const string DisasmSha = "3c57260a519cf288bee0f13b09e5a8fc2d3d30d095aa8549e6017d181a2b2343";
    const string PdbGuid = "1797f1f9-ec73-4291-98a8-5c9c8aaf2ae1";
    const string ImageSha = "8de750766ea4026303f7d3b8f0c1ce91da67b12b4af293245cf24faa98d3a3ac";
    const ulong PreferredImageBase = 0x140000000, HeadStart = 0xb13f0, HeadEnd = 0xb1d36;
    const int TargetPid = 79568, TargetIndex = 557;
    static readonly JavaScriptSerializer Json = new JavaScriptSerializer { MaxJsonLength = Int32.MaxValue, RecursionLimit = 100 };
    static readonly Dictionary<string, string> Bound = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    static readonly Dictionary<string, string> Dependencies = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    static Dictionary<string, object> Obj(params object[] p) { var d = new Dictionary<string, object>(); for (int i = 0; i < p.Length; i += 2) d.Add((string)p[i], p[i + 1]); return d; }
    static Dictionary<string, object> Dict(object x) { return (Dictionary<string, object>)x; }
    static string Str(object x) { return Convert.ToString(x, CultureInfo.InvariantCulture); }
    static long Num(object x) { return Convert.ToInt64(x, CultureInfo.InvariantCulture); }
    static double Real(object x) { return Convert.ToDouble(x, CultureInfo.InvariantCulture); }
    static void Need(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static string HashBytes(byte[] b) { using (var h = SHA256.Create()) return BitConverter.ToString(h.ComputeHash(b)).Replace("-", "").ToLowerInvariant(); }
    static string Hash(string p) { using (var f = new FileStream(p, FileMode.Open, FileAccess.Read, FileShare.Read)) using (var h = SHA256.Create()) return BitConverter.ToString(h.ComputeHash(f)).Replace("-", "").ToLowerInvariant(); }
    static Dictionary<string, object> ReadBound(string path, string sha)
    {
        path = Path.GetFullPath(path); byte[] b = File.ReadAllBytes(path);
        Need(HashBytes(b) == sha, "Input digest differs: " + path); Bound.Add(path, sha);
        return Dict(Json.DeserializeObject(Encoding.UTF8.GetString(b)));
    }
    static bool SamePath(string a, string b) { return String.Equals(Path.GetFullPath(a), Path.GetFullPath(b), StringComparison.OrdinalIgnoreCase); }
    static bool InHead(ulong rva) { return rva >= HeadStart && rva < HeadEnd; }
    static string Bucket(bool owner, double time, double start, double stop, double sessionStop)
    {
        if (!owner) return "wrong_owner";
        if (Double.IsNaN(time) || Double.IsInfinity(time) || time < 0 || time > sessionStop || time < start || time > stop) return "outside";
        return "in_bounds";
    }
    static void Add(Dictionary<ulong, long> d, ulong key) { long n; d.TryGetValue(key, out n); d[key] = n + 1; }
    static object Histogram(Dictionary<ulong, long> d) { return d.OrderBy(x => x.Key).Select(x => Obj("rva_hex", "0x" + x.Key.ToString("x"), "sample_count", x.Value)).ToList(); }
    static Dictionary<string, string> Parse(string[] args)
    {
        var names = new[] { "--audit", "--receipt", "--disassembly", "--build", "--build-sha256", "--output" };
        Need(args.Length == names.Length * 2, "Six named argument/value pairs required");
        var d = new Dictionary<string, string>();
        for (int i = 0; i < args.Length; i += 2) { Need(names.Contains(args[i]) && !d.ContainsKey(args[i]), "Unknown/repeated argument"); d.Add(args[i], args[i + 1]); }
        Need(names.All(d.ContainsKey), "Missing argument"); return d;
    }
    public static int Main(string[] args)
    {
        if (args.Length == 1 && args[0] == "--self-test") return SelfTest();
        try
        {
            var options = Parse(args);
            var audit = ReadBound(options["--audit"], AuditSha);
            // Use only the four dependencies already observed and hashed by the pinned audit.
            foreach (object item in (object[])audit["loaded_dependencies"])
            {
                var row = Dict(item); string path = Str(row["path"]), sha = Str(row["sha256_before"]);
                Need((bool)row["unchanged"] && sha == Str(row["sha256_after"]) && Hash(path) == sha, "Dependency differs: " + path);
                Dependencies.Add(Path.GetFullPath(path), sha);
            }
            AppDomain.CurrentDomain.AssemblyResolve += delegate(object sender, ResolveEventArgs request)
            {
                string filename = new AssemblyName(request.Name).Name + ".dll";
                var matches = Dependencies.Keys.Where(p => String.Equals(Path.GetFileName(p), filename, StringComparison.OrdinalIgnoreCase)).ToList();
                if (matches.Count != 1) return null;
                Need(Hash(matches[0]) == Dependencies[matches[0]], "Dependency changed before load"); return Assembly.LoadFrom(matches[0]);
            };
            return Run(options, audit);
        }
        catch (Exception e) { Console.Error.WriteLine(e); return 2; }
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static int Run(Dictionary<string, string> args, Dictionary<string, object> audit)
    {
        string output = Path.GetFullPath(args["--output"]);
        Need(!File.Exists(output) && Directory.Exists(Path.GetDirectoryName(output)), "Fresh output in existing directory required");
        var report = Obj("schema_version", 1, "kind", "compact-head-instruction-histogram-v1", "status", "started", "arguments", args,
            "started_utc", DateTime.UtcNow.ToString("O"), "strict_profile_status", audit["status"], "strict_profile_verdict_unchanged", true,
            "no_symbols_or_conversion_or_model", true, "source_and_input_closure", false,
            "limitations", new[] { "Exploratory addresses only; original strict full-profile audit remains failed.",
                "Leaf samples are instruction addresses, not retired instruction/cycle counts or bandwidth measurements; sampling skid is not quantified.",
                "External-leaf head-frame RVAs are saved TraceEvent code addresses, not verified call instruction addresses; no return-address adjustment or callee attribution is performed.",
                "ETLX is retained under a read lock with size/write-time checks, not rehashed here. Its digest is inherited from the pinned audit.",
                "The disassembly has no raw instruction lengths. Address annotation coverage is not claimed; unmapped RVAs remain raw.",
                "No complete stack unwind or symbol completeness claim. Whole-process samples have no exact prefill/decode markers." });
        int exit = 2;
        using (var destination = new FileStream(output, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
        {
            try
            {
                var build = ReadBound(args["--build"], args["--build-sha256"]);
                Need(Str(build["kind"]) == "compact-ip-histogram-build-v1" && Num(build["compile_exit_code"]) == 0 && build["host_self_test_exit_code"] != null && Num(build["host_self_test_exit_code"]) == 0 && (bool)build["source_window_unchanged"], "Build rejected");
                string executable = Assembly.GetExecutingAssembly().Location;
                Need(Hash(executable) == Str(build["compiled_executable_sha256"]), "Executing binary differs from build");
                Bound.Add(executable, Str(build["compiled_executable_sha256"]));
                string source = Str(build["archived_source"]), sourceSha = Str(build["archived_source_sha256"]);
                Need(Hash(source) == sourceSha, "Archived source differs"); Bound.Add(source, sourceSha);
                report["helper"] = Obj("path", executable, "sha256", Bound[executable], "source", source, "source_sha256", sourceSha);
                var receipt = ReadBound(args["--receipt"], ReceiptSha);
                Need((bool)receipt["source_and_input_closure"] && (bool)receipt["pdb_identity_matches_pe"], "Symbol receipt is not closed");
                Need(Str(receipt["function"]) == "falcon_ocr::kernels::attention64_candidate::compact_head" && Str(receipt["virtual_address"]) == "0x1400b13f0" && Str(receipt["stop_address"]) == "0x1400b1d36" && Num(receipt["code_bytes"]) == (long)(HeadEnd - HeadStart), "Function extent differs");
                var cv = Dict(receipt["pe_codeview"]);
                Need(new Guid(Str(cv["guid"])) == new Guid(PdbGuid) && Num(cv["age"]) == 1, "Receipt PDB identity differs");
                string disasmPath = Path.GetFullPath(args["--disassembly"]);
                byte[] disasmBytes = File.ReadAllBytes(disasmPath);
                Need(HashBytes(disasmBytes) == DisasmSha && Str(Dict(receipt["output_sha256"])["head-disassembly.txt"]) == DisasmSha, "Disassembly differs"); Bound.Add(disasmPath, DisasmSha);
                var targetMeta = Dict(audit["target"]); var sampleMeta = Dict(audit["samples"]); var retained = Dict(audit["converted_etlx"]);
                Need(Num(audit["target_pid"]) == TargetPid && Num(targetMeta["process_index"]) == TargetIndex && Str(audit["status"]) == "audit_failed", "Wrong fixed audit");
                string imagePath = Dict(receipt["input_sha256"]).Single(x => Str(x.Value) == ImageSha).Key;
                string etlx = Str(retained["path"]); var info = new FileInfo(etlx); long size = info.Length; DateTime write = info.LastWriteTimeUtc;
                Need(size == Num(retained["bytes"]), "Retained ETLX size differs");
                report["retained_etlx"] = Obj("path", etlx, "bytes", size, "sha256_from_audit", retained["sha256"], "rehashed", false);
                using (var held = new FileStream(etlx, FileMode.Open, FileAccess.Read, FileShare.Read))
                using (var log = new TraceLog(etlx))
                {
                    var actualLibrary = typeof(TraceLog).Assembly;
                    Need(Dependencies.ContainsKey(actualLibrary.Location) && Hash(actualLibrary.Location) == Dependencies[actualLibrary.Location], "Loaded TraceEvent differs");
                    var targets = log.Processes.Where(p => p.ProcessID == TargetPid).ToList(); Need(targets.Count == 1, "PID missing/reused"); var target = targets[0];
                    Need((int)target.ProcessIndex == TargetIndex && target.CommandLine == Str(targetMeta["command_line"]) && target.StartTimeRelativeMsec == Real(targetMeta["start_relative_ms"]) && target.EndTimeRelativeMsec == Real(targetMeta["end_relative_ms"]), "Target identity/lifetime differs");
                    var modules = target.LoadedModules.Where(m => SamePath(m.FilePath, imagePath)).ToList(); Need(modules.Count == 1, "Missing/ambiguous project module"); var module = modules[0];
                    var auditModules = ((object[])audit["modules"]).Select(Dict).Where(m => SamePath(Str(m["path"]), imagePath)).ToList(); Need(auditModules.Count == 1, "Audit module ambiguous"); var am = auditModules[0];
                    Need(module.ModuleFile.PdbSignature == new Guid(PdbGuid) && module.ModuleFile.PdbAge == 1 && module.ImageBase == Convert.ToUInt64(Str(am["load_base_hex"]).Substring(2), 16) && module.ModuleFile.ImageSize == Num(am["image_size"]) && (int)module.ModuleFile.ModuleFileIndex == Num(am["module_file_index"]) && module.LoadTimeRelativeMSec == Real(am["load_relative_ms"]) && module.UnloadTimeRelativeMSec == Real(am["unload_relative_ms"]), "Module/PDB/layout identity differs");
                    Need(HeadEnd <= (ulong)module.ModuleFile.ImageSize, "Head outside module");
                    report["module"] = Obj("path", module.FilePath, "image_sha256_from_closed_receipt", ImageSha, "runtime_base_hex", "0x" + module.ImageBase.ToString("x"), "preferred_image_base_hex", "0x" + PreferredImageBase.ToString("x"), "pdb_guid", PdbGuid, "pdb_age", 1, "head_start_rva_hex", "0x" + HeadStart.ToString("x"), "head_end_exclusive_rva_hex", "0x" + HeadEnd.ToString("x"), "load_ms", module.LoadTimeRelativeMSec, "unload_ms", module.UnloadTimeRelativeMSec);
                    var leaves = new Dictionary<ulong, long>(); var externalFrames = new Dictionary<ulong, long>();
                    var flaggedLeaves = new Dictionary<ulong, long>(); var flaggedExternalFrames = new Dictionary<ulong, long>();
                    var exceptions = new List<object>(); var classes = new Dictionary<string, long[]>();
                    foreach (string name in new[] { "wrong_owner", "outside", "in_bounds" }) classes.Add(name, new long[5]);
                    long examined = 0, stacked = 0, externalNoStack = 0, externalStacked = 0, externalWithHead = 0, multipleHeadFrames = 0, headLeafOutsideModuleLifetime = 0;
                    var watch = Stopwatch.StartNew();
                    foreach (var ev in log.Events)
                    {
                        examined++; if ((examined & 16383) == 0) Need(examined <= 20000000 && watch.Elapsed.TotalSeconds <= 600, "Bounded scan exceeded20M events/600s");
                        var sample = ev as SampledProfileTraceData; if (sample == null || sample.ProcessID != TargetPid) continue;
                        var owner = sample.Process(); double time = sample.TimeStampRelativeMSec;
                        string kind = Bucket(owner != null && (int)owner.ProcessIndex == TargetIndex, time, target.StartTimeRelativeMsec, target.EndTimeRelativeMsec, log.SessionEndTimeRelativeMSec);
                        long[] counts = classes[kind]; counts[0]++; counts[1] += sample.Count; if (sample.ExecutingDPC) counts[2]++; if (sample.ExecutingISR) counts[3]++; if (sample.NonProcess) counts[4]++;
                        if (kind != "in_bounds")
                        {
                            Need(exceptions.Count < 32, "More than32 exceptional target samples");
                            exceptions.Add(Obj("classification", kind, "event_index", (uint)sample.EventIndex, "relative_ms", Double.IsNaN(time) || Double.IsInfinity(time) ? null : (object)time, "qpc", sample.TimeStampQPC, "thread_id", sample.ThreadID, "owner_process_index", owner == null ? null : (object)(int)owner.ProcessIndex, "ip_hex", "0x" + sample.InstructionPointer.ToString("x"), "dpc", sample.ExecutingDPC, "isr", sample.ExecutingISR, "non_process", sample.NonProcess)); continue;
                        }
                        bool hasStack = log.GetCallStackIndexForEvent(sample) != CallStackIndex.Invalid; if (hasStack) stacked++;
                        ulong ip = sample.InstructionPointer; bool flagged = sample.ExecutingDPC || sample.ExecutingISR || sample.NonProcess;
                        if (ip >= module.ImageBase && InHead(ip - module.ImageBase))
                        {
                            ulong rva = ip - module.ImageBase; Add(leaves, rva); if (flagged) Add(flaggedLeaves, rva);
                            if (time < module.LoadTimeRelativeMSec || time > module.UnloadTimeRelativeMSec) headLeafOutsideModuleLifetime++;
                            continue;
                        }
                        if (!hasStack) { externalNoStack++; continue; } externalStacked++;
                        var frame = log.GetCallStackForEvent(sample); Need(frame != null, "Valid stack index but missing stack");
                        var seen = new HashSet<ulong>(); int depth = 0;
                        while (frame != null)
                        {
                            Need(++depth <= 256, "Stack exceeds bounded256 frames"); var address = frame.CodeAddress; ulong raw = address.Address;
                            if (raw >= module.ImageBase && InHead(raw - module.ImageBase))
                            {
                                Need(!String.IsNullOrEmpty(address.ModuleFilePath) && SamePath(address.ModuleFilePath, imagePath), "Head-range stack address has wrong/unknown module");
                                seen.Add(raw - module.ImageBase);
                            }
                            frame = frame.Caller;
                        }
                        if (seen.Count > 0) externalWithHead++; if (seen.Count > 1) multipleHeadFrames++;
                        foreach (ulong rva in seen) { Add(externalFrames, rva); if (flagged) Add(flaggedExternalFrames, rva); }
                    }
                    Need(examined <= 20000000 && watch.Elapsed.TotalSeconds <= 600, "Bounded scan exceeded20M events/600s");
                    report["all_target_classifications"] = classes.ToDictionary(x => x.Key, x => (object)Obj("samples", x.Value[0], "count_payload_total", x.Value[1], "dpc", x.Value[2], "isr", x.Value[3], "non_process", x.Value[4]));
                    report["exceptional_samples"] = exceptions;
                    report["counts"] = Obj("all_events_examined", examined, "in_bounds_samples", classes["in_bounds"][0], "with_attached_stack", stacked, "head_leaf_samples", leaves.Values.Sum(), "head_leaf_outside_module_lifetime", headLeafOutsideModuleLifetime, "external_leaf_without_stack", externalNoStack, "external_leaf_with_stack", externalStacked, "external_leaf_with_head_frame", externalWithHead, "external_leaf_with_multiple_distinct_head_frame_rvas", multipleHeadFrames);
                    report["head_leaf_rvas"] = Histogram(leaves); report["flagged_head_leaf_rvas"] = Histogram(flaggedLeaves);
                    report["external_leaf_head_frame_rvas"] = Histogram(externalFrames); report["flagged_external_leaf_head_frame_rvas"] = Histogram(flaggedExternalFrames);
                    report["histogram_semantics"] = "Head leaf and external leaf populations are disjoint. Flags are reported, never filtered. Each external sample contributes at most once per distinct saved head-frame RVA; several distinct RVAs may contribute, so frame-histogram totals may exceed samples. No address adjustment or function subdivision.";
                    Need(classes["in_bounds"][0] == Num(sampleMeta["event_count"]) && classes["in_bounds"][1] == Num(sampleMeta["count_payload_total"]) && stacked == Num(sampleMeta["with_attached_stack"]) && classes["wrong_owner"][0] == Num(sampleMeta["wrong_or_missing_process_index_count"]) && classes["outside"][0] == Num(sampleMeta["out_of_bounds_count"]) && classes["in_bounds"][2] == Num(sampleMeta["dpc_count"]) && classes["in_bounds"][3] == Num(sampleMeta["isr_count"]) && classes["in_bounds"][4] == Num(sampleMeta["non_process_count"]), "Sample accounting differs from original audit");
                    Need(leaves.Values.Sum() + externalNoStack + externalStacked == classes["in_bounds"][0] && headLeafOutsideModuleLifetime == 0, "Head accounting/module lifetime differs");
                    info.Refresh(); Need(info.Length == size && info.LastWriteTimeUtc == write, "Retained ETLX metadata changed");
                    report["retained_read_lock_and_metadata_unchanged"] = true;
                }
                foreach (var p in Bound) Need(Hash(p.Key) == p.Value, "Bound input/source changed: " + p.Key);
                foreach (var p in Dependencies) Need(Hash(p.Key) == p.Value, "Dependency changed: " + p.Key);
                var loadedDependencies = new List<object>();
                foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
                {
                    if (assembly.IsDynamic || String.IsNullOrEmpty(assembly.Location)) continue;
                    string name = Path.GetFileName(assembly.Location);
                    var expected = Dependencies.Keys.Where(p => String.Equals(Path.GetFileName(p), name, StringComparison.OrdinalIgnoreCase)).ToList();
                    if (expected.Count == 0) continue;
                    Need(expected.Count == 1 && SamePath(assembly.Location, expected[0]) && Hash(assembly.Location) == Dependencies[expected[0]], "Actual loaded dependency differs: " + name);
                    loadedDependencies.Add(Obj("assembly", assembly.FullName, "path", assembly.Location, "sha256", Dependencies[expected[0]]));
                }
                report["loaded_dependencies"] = loadedDependencies;
                report["source_and_input_closure"] = true; report["status"] = "diagnostic_complete_strict_profile_failed"; exit = 0;
            }
            catch (Exception e) { report["status"] = "failed"; report["error"] = e.ToString(); exit = 2; }
            finally
            {
                report["bound_small_inputs"] = Bound; report["dependency_sha256"] = Dependencies;
                report["ended_utc"] = DateTime.UtcNow.ToString("O"); report["exit_code"] = exit;
                byte[] b = Encoding.UTF8.GetBytes(Json.Serialize(report) + "\n"); destination.Write(b, 0, b.Length); destination.Flush(true);
            }
        }
        Console.WriteLine("{0}: {1}", report["status"], output); return exit;
    }
    static int SelfTest()
    {
        int n = 0;
        foreach (ulong rva in new[] { HeadStart, HeadEnd - 1 }) { Need(InHead(rva), "Head endpoint rejected"); n++; }
        foreach (ulong rva in new[] { HeadStart - 1, HeadEnd, UInt64.MaxValue }) { Need(!InHead(rva), "Out-of-head address accepted"); n++; }
        Need(Bucket(true, 10, 10, 20, 30) == "in_bounds" && Bucket(true, 20, 10, 20, 30) == "in_bounds", "Inclusive lifecycle bounds"); n++;
        foreach (double time in new[] { 9.0, 21.0, -1.0, Double.NaN, Double.PositiveInfinity }) { Need(Bucket(true, time, 10, 20, 30) == "outside", "Outside sample accepted"); n++; }
        Need(Bucket(false, 15, 10, 20, 30) == "wrong_owner" && Bucket(false, Double.NaN, 10, 20, 30) == "wrong_owner", "Ownership precedence differs"); n++;
        Need(Bucket(true, 25, 10, 30, 20) == "outside", "Session bound ignored"); n++;
        var bins = new Dictionary<ulong, long>(); Add(bins, HeadStart); Add(bins, HeadStart); Need(bins[HeadStart] == 2, "Histogram loss"); n++;
        Console.WriteLine("CompactIpHistogram pure host self-tests passed: " + n); return 0;
    }
}
