import { afterEach, describe, expect, it, vi } from "vitest"

import { API_BASE, clearSessionBeacon, sendQuery, uploadDocument } from "@/lib/api"

afterEach(() => vi.unstubAllGlobals())

describe("clearSessionBeacon", () => {
  it("clears only this session, never the whole database", () => {
    const beacon = vi.fn(() => true)
    vi.stubGlobal("navigator", { sendBeacon: beacon })
    clearSessionBeacon("abc 1")
    expect(beacon).toHaveBeenCalledOnce()
    const [url] = beacon.mock.calls[0] as unknown as [string]
    expect(url).toBe(`${API_BASE}/api/sessions/abc%201/clear`)
    expect(url).not.toContain("/api/reset")
  })
})

describe("sendQuery", () => {
  it("sends document scope only when given", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ answer: "a", route: "RAG" }), { status: 200 }),
    )
    vi.stubGlobal("fetch", fetchMock)

    const result = await sendQuery("q", "s", { documentIds: ["d1"] })
    const body = JSON.parse((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body as string)
    expect(body).toEqual({ query: "q", session_id: "s", document_ids: ["d1"] })
    // Normalised even when the backend omits them.
    expect(result.artifacts).toEqual([])
    expect(result.segments).toEqual([])
  })
})

describe("uploadDocument", () => {
  it("throws the backend's reason for a failed ingestion", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ error: "OCR returned no pages", code: "ingest_failed" }), { status: 422 }),
      ),
    )
    await expect(uploadDocument(new File(["%PDF-"], "a.pdf"))).rejects.toThrow("OCR returned no pages")
  })

  it("resolves a duplicate as skipped", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ skipped: true, document_id: null }), { status: 200 })),
    )
    expect((await uploadDocument(new File(["%PDF-"], "a.pdf"))).skipped).toBe(true)
  })
})
