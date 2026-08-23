// GhostType parse oracle: JSON-lines stdin/stdout service over ScriptDom.
// Ops:
//   {"op":"version"}
//   {"op":"parse","sql":"..."}                          -> errors for a document
//   {"op":"insertDelta","prefix":"","candidate":"","suffix":""}
//        -> baseline errors (prefix+suffix) vs candidate errors
//           (prefix+candidate+suffix); introduced = candidate - baseline.
// Uses TSql170Parser (SQL Server 2025 syntax), QUOTED_IDENTIFIER ON.
using System.Reflection;
using System.Text.Json;
using Microsoft.SqlServer.TransactSql.ScriptDom;

var parserVersion = typeof(TSql170Parser).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
    ?? typeof(TSql170Parser).Assembly.GetName().Version?.ToString() ?? "unknown";

string? line;
var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
while ((line = Console.ReadLine()) is not null)
{
    if (string.IsNullOrWhiteSpace(line)) continue;
    try
    {
        var doc = JsonDocument.Parse(line);
        var op = doc.RootElement.GetProperty("op").GetString();
        object result;
        switch (op)
        {
            case "version":
                result = new { ok = true, parser = "TSql170Parser", assemblyVersion = parserVersion };
                break;
            case "parse":
            {
                var sql = doc.RootElement.GetProperty("sql").GetString() ?? "";
                var parsed = Doc.Parse(sql);
                result = new
                {
                    ok = true,
                    errorCount = parsed.Errors.Count,
                    structuralErrorCount = parsed.Errors.Count(e => !e.Trailing),
                    lastTokenOffset = parsed.LastTokenOffset,
                    errors = parsed.Errors.Select(Doc.Shape).ToList(),
                };
                break;
            }
            case "insertDelta":
            {
                var prefix = doc.RootElement.GetProperty("prefix").GetString() ?? "";
                var candidate = doc.RootElement.GetProperty("candidate").GetString() ?? "";
                var suffix = doc.RootElement.GetProperty("suffix").GetString() ?? "";
                var baselineDoc = Doc.Parse(prefix + suffix);
                var candidateDoc = Doc.Parse(prefix + candidate + suffix);
                var remaining = Doc.StructuralMultiset(baselineDoc);
                var structuralIntroduced = new List<object>();
                foreach (var e in candidateDoc.Errors)
                {
                    if (e.Trailing) continue;
                    var key = $"{e.Number}:{e.Message}";
                    if (remaining.GetValueOrDefault(key) > 0) remaining[key] -= 1;
                    else structuralIntroduced.Add(Doc.Shape(e));
                }
                result = new
                {
                    ok = true,
                    baselineErrorCount = baselineDoc.Errors.Count,
                    baselineStructuralCount = baselineDoc.Errors.Count(e => !e.Trailing),
                    candidateErrorCount = candidateDoc.Errors.Count,
                    candidateStructuralCount = candidateDoc.Errors.Count(e => !e.Trailing),
                    candidateTrailingCount = candidateDoc.Errors.Count(e => e.Trailing),
                    lastTokenOffset = candidateDoc.LastTokenOffset,
                    structuralIntroducedCount = structuralIntroduced.Count,
                    structuralIntroduced,
                    candidateErrors = candidateDoc.Errors.Select(Doc.Shape).ToList(),
                };
                break;
            }
            default:
                result = new { ok = false, error = $"unknown op {op}" };
                break;
        }
        Console.WriteLine(JsonSerializer.Serialize(result));
    }
    catch (Exception ex)
    {
        Console.WriteLine(JsonSerializer.Serialize(new { ok = false, error = ex.Message }));
    }
}

// Ghost-text documents are usually incomplete statements (582/588 dataset
// cases have an empty suffix), so parse errors are classified by position:
//   trailing  — error 46029 (unexpected EOF) or an error at/after the start
//               of the last meaningful token: expected tail incompleteness.
//   structural — an error strictly before the last meaningful token: a real
//               syntax problem inside the typed/inserted region.
record ParsedDoc(List<ErrorInfo> Errors, int LastTokenOffset);
record ErrorInfo(int Number, int Line, int Column, int Offset, string Message, bool Trailing);

static class Doc
{
    public static ParsedDoc Parse(string sql)
    {
        var parser = new TSql170Parser(initialQuotedIdentifiers: true);
        using var reader = new StringReader(sql);
        var fragment = parser.Parse(reader, out IList<ParseError> errors);
        var lastTokenOffset = 0;
        if (fragment?.ScriptTokenStream is not null)
        {
            foreach (var token in fragment.ScriptTokenStream)
            {
                if (token.TokenType is TSqlTokenType.EndOfFile or TSqlTokenType.WhiteSpace
                    or TSqlTokenType.SingleLineComment or TSqlTokenType.MultilineComment) continue;
                if (token.Offset > lastTokenOffset) lastTokenOffset = token.Offset;
            }
        }
        var infos = new List<ErrorInfo>();
        foreach (var e in errors)
        {
            var trailing = e.Number == 46029 || e.Offset >= lastTokenOffset;
            infos.Add(new ErrorInfo(e.Number, e.Line, e.Column, e.Offset, e.Message, trailing));
        }
        return new ParsedDoc(infos, lastTokenOffset);
    }

    // Multiset of number:message keys over structural errors only; position-
    // independent so unchanged context errors cancel, but a second occurrence
    // of the same message still counts as introduced.
    public static Dictionary<string, int> StructuralMultiset(ParsedDoc doc)
    {
        var keys = new Dictionary<string, int>();
        foreach (var e in doc.Errors)
        {
            if (e.Trailing) continue;
            var key = $"{e.Number}:{e.Message}";
            keys[key] = keys.GetValueOrDefault(key) + 1;
        }
        return keys;
    }

    public static object Shape(ErrorInfo e) =>
        new { number = e.Number, line = e.Line, column = e.Column, offset = e.Offset, message = e.Message, trailing = e.Trailing };
}
