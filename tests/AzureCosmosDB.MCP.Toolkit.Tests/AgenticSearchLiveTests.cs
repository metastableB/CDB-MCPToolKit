using System.Net.Http.Json;
using System.Text.Json;
using FluentAssertions;
using Xunit;
using Xunit.Abstractions;

namespace AzureCosmosDB.MCP.Toolkit.Tests;

/// <summary>Call the Python agent through both MCP transports using real model and Cosmos backends.</summary>
[Collection("Live agentic search")]
public sealed class AgenticSearchLiveTests : IDisposable
{
    private readonly McpTestApplicationFactory _factory;
    private readonly HttpClient _client;
    private readonly ITestOutputHelper _output;
    private readonly string _database;
    private string? _sessionId;
    private int _requestId;

    public AgenticSearchLiveTests(ITestOutputHelper output)
    {
        var url = Environment.GetEnvironmentVariable("COSMOS_RETRIEVER_URL");
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) ||
            uri.Scheme != Uri.UriSchemeHttp || !uri.IsLoopback)
            throw new InvalidOperationException("Live MCP tests require a loopback COSMOS_RETRIEVER_URL.");
        _database = Environment.GetEnvironmentVariable("COSMOS_TEST_DATABASE") ?? "mcp-live-tests-v1";
        if (!_database.StartsWith("mcp-live-tests-", StringComparison.Ordinal))
            throw new InvalidOperationException("Live MCP tests require a prepared mcp-live-tests- database.");
        _output = output;
        _factory = new McpTestApplicationFactory();
        _client = _factory.CreateClient();
        _client.Timeout = TimeSpan.FromMinutes(5);
    }

    [LiveMcpTheory]
    [InlineData("/mcp", null)]
    [InlineData("/mcp/http", null)]
    [InlineData("/mcp", "scifact-100-v1")]
    [InlineData("/mcp/http", "scifact-100-v1")]
    public async Task AgenticSearch_returns_answer_and_evidence(string transport, string? container)
    {
        await InitializeAsync(transport);
        var result = await SearchAsync(transport,
            "Search for documents about vitamin B12 and homocysteine. Summarize the evidence, distinguishing direct from indirect evidence.",
            container);
        AssertAnswer(result, container is null
            ? new[] { "flat-v1", "nested-v1", "fields-v1", "scifact-100-v1" }
            : new[] { container });
    }

    [LiveMcpTheory]
    [InlineData("/mcp", "container")]
    [InlineData("/mcp/http", "container")]
    [InlineData("/mcp", "database")]
    [InlineData("/mcp/http", "database")]
    public async Task AgenticSearch_rejects_unknown_scope(string transport, string field)
    {
        await InitializeAsync(transport);
        var result = await SearchAsync(transport, "Find battery recycling documents.",
            field == "container" ? "not-configured" : null,
            field == "database" ? "not-configured" : null);
        result.GetProperty("error").GetString().Should().Contain("configured");
    }

    [LiveMcpTheory]
    [InlineData("/mcp")]
    [InlineData("/mcp/http")]
    public async Task Concurrent_agent_requests_keep_separate_evidence(string transport)
    {
        await InitializeAsync(transport);
        var containers = new[] { "flat-v1", "nested-v1" };
        var results = await Task.WhenAll(containers.Select(container => SearchAsync(transport,
            "Search for battery recycling. Which canary marker words appear in the documents? Answer from the retrieved text.",
            container)));
        for (var index = 0; index < results.Length; index++)
            AssertAnswer(results[index], new[] { containers[index] });
        var firstIds = results[0].GetProperty("documents").EnumerateArray()
            .Select(item => item.GetProperty("retrieval_id").GetString()).ToHashSet();
        results[1].GetProperty("documents").EnumerateArray()
            .Select(item => item.GetProperty("retrieval_id").GetString())
            .Should().NotIntersectWith(firstIds);
    }

    private void AssertAnswer(JsonElement result, string[] containers)
    {
        result.TryGetProperty("error", out _).Should().BeFalse(result.ToString());
        result.GetProperty("answer").GetString().Should().NotBeNullOrWhiteSpace();
        result.GetProperty("terminal_reason").GetString().Should().Be("stop");
        result.GetProperty("turns").GetInt32().Should().BeInRange(2, 6);
        result.GetProperty("errors").GetArrayLength().Should().Be(0);
        result.GetProperty("partial").GetBoolean().Should().BeFalse();
        result.GetProperty("searched").EnumerateArray()
            .Select(target => target.GetProperty("container").GetString())
            .Should().BeEquivalentTo(containers);
        var documents = result.GetProperty("documents").EnumerateArray().ToArray();
        documents.Length.Should().BeInRange(1, 5);
        documents.Select(item => item.GetProperty("retrieval_id").GetString()).Should().OnlyHaveUniqueItems();
        documents.Select(item => item.GetProperty("rank").GetInt32())
            .Should().Equal(Enumerable.Range(0, documents.Length));
        foreach (var document in documents)
        {
            document.GetProperty("database").GetString().Should().Be(_database);
            containers.Should().Contain(document.GetProperty("container").GetString());
            document.GetProperty("text").GetString().Should().NotBeNullOrWhiteSpace();
            document.GetProperty("cosmos_identity").GetProperty("id").GetString().Should().NotBeNullOrWhiteSpace();
        }
    }

    private async Task InitializeAsync(string transport)
    {
        await RpcAsync(transport, "initialize", new
        {
            protocolVersion = "2025-03-26", capabilities = new { },
            clientInfo = new { name = "live-agent-tests", version = "1.0" }
        });
        if (transport == "/mcp")
        {
            using var request = MakeRequest(transport, new { jsonrpc = "2.0", method = "notifications/initialized" });
            using var response = await _client.SendAsync(request);
            response.EnsureSuccessStatusCode();
        }
        var tools = await RpcAsync(transport, "tools/list", new { });
        tools.GetProperty("tools").EnumerateArray()
            .Select(tool => tool.GetProperty("name").GetString()).Should().Contain("agentic_search");
    }

    private async Task<JsonElement> SearchAsync(string transport, string query, string? container, string? database = null)
    {
        var arguments = new Dictionary<string, object>
        {
            ["query"] = query, ["maxDocuments"] = 5, ["database"] = database ?? _database
        };
        if (container is not null) arguments["container"] = container;
        var result = await RpcAsync(transport, "tools/call", new { name = "agentic_search", arguments });
        using var payload = JsonDocument.Parse(result.GetProperty("content")[0].GetProperty("text").GetString()!);
        _output.WriteLine(JsonSerializer.Serialize(new { transport, arguments, response = payload.RootElement }));
        return payload.RootElement.Clone();
    }

    private HttpRequestMessage MakeRequest(string transport, object payload)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, transport) { Content = JsonContent.Create(payload) };
        request.Headers.TryAddWithoutValidation("Accept", "application/json, text/event-stream");
        if (transport == "/mcp" && _sessionId is not null)
        {
            request.Headers.TryAddWithoutValidation("Mcp-Session-Id", _sessionId);
            request.Headers.TryAddWithoutValidation("MCP-Protocol-Version", "2025-03-26");
        }
        return request;
    }

    private async Task<JsonElement> RpcAsync(string transport, string method, object parameters)
    {
        using var request = MakeRequest(transport, new
        {
            jsonrpc = "2.0", id = Interlocked.Increment(ref _requestId), method, @params = parameters
        });
        using var response = await _client.SendAsync(request);
        response.EnsureSuccessStatusCode();
        if (response.Headers.TryGetValues("Mcp-Session-Id", out var sessions)) _sessionId = sessions.First();
        var body = await response.Content.ReadAsStringAsync();
        if (response.Content.Headers.ContentType?.MediaType == "text/event-stream")
            body = body.Split('\n').First(line => line.StartsWith("data: ", StringComparison.Ordinal))[6..];
        using var envelope = JsonDocument.Parse(body);
        return envelope.RootElement.GetProperty("result").Clone();
    }

    public void Dispose()
    {
        _client.Dispose();
        _factory.Dispose();
    }
}

public sealed class LiveMcpTheoryAttribute : TheoryAttribute
{
    public LiveMcpTheoryAttribute()
    {
        if (Environment.GetEnvironmentVariable("RUN_COSMOS_MCP_LIVE") != "1")
            Skip = "Run the opt-in Python live-test launcher to start the retriever service.";
    }
}

[CollectionDefinition("Live agentic search", DisableParallelization = true)]
public sealed class LiveAgenticSearchCollection;