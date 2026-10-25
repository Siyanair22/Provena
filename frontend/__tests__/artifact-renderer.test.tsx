import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

import { ArtifactRenderer, ChartBoundary } from "@/components/artifact-renderer"
import type { Artifact } from "@/lib/types"

function artifact(overrides: Partial<Artifact> = {}): Artifact {
  return {
    id: "art_0",
    schema_version: 1,
    kind: "table",
    type: "table",
    title: "Revenue by segment",
    data: [
      { label: "Enterprise", value: 1204.5 },
      { label: "SMB", value: 88 },
    ],
    encoding: { x: null, y: [], series: null },
    columns: [
      { key: "label", label: "Segment", kind: "categorical", align: "left" },
      { key: "value", label: "Revenue", kind: "numeric", align: "right", decimals: 1 },
    ],
    unit: null,
    provenance: {
      sql: "SELECT 1",
      tables_used: ["t-1"],
      row_count: 2,
      data_rows: 2,
      truncated: false,
      partial: false,
      source_documents: [],
      answer_is_llm_merged: false,
    },
    notes: [],
    classifier: { rule: "R14_default", version: 1, pivoted: false },
    ...overrides,
  }
}

describe("ArtifactRenderer", () => {
  it("renders a table with its title and formatted values", () => {
    render(<ArtifactRenderer artifact={artifact()} />)
    expect(screen.getByText("Revenue by segment")).toBeTruthy()
    expect(screen.getByText("Segment")).toBeTruthy()
    expect(screen.getByText("Enterprise")).toBeTruthy()
    expect(screen.getByText(/1,204\.5/)).toBeTruthy()
    expect(screen.getByText("2 rows")).toBeTruthy()
  })

  it("renders a scalar with its unit", () => {
    render(
      <ArtifactRenderer
        artifact={artifact({
          kind: "scalar",
          type: "scalar",
          value: 42,
          unit: "%",
          data: [{ result: 42 }],
          columns: [{ key: "result", label: "result", kind: "numeric", decimals: 0 }],
        })}
      />,
    )
    expect(screen.getByText("42%")).toBeTruthy()
  })

  it("says so when the query matched nothing", () => {
    render(<ArtifactRenderer artifact={artifact({ kind: "empty", type: "empty", data: [] })} />)
    expect(screen.getByText(/matched no rows/)).toBeTruthy()
  })

  it("prints the partial-result caveat once, from the backend's notes", () => {
    const note = "Result hit the query limit; more rows may exist."
    const { container } = render(
      <ArtifactRenderer
        artifact={artifact({
          notes: [note],
          provenance: { ...artifact().provenance, partial: true },
        })}
      />,
    )
    expect(container.textContent?.split(note).length).toBe(2)
  })

  it("labels a merged answer's chart as the exact part", () => {
    render(
      <ArtifactRenderer
        artifact={artifact({
          provenance: { ...artifact().provenance, answer_is_llm_merged: true },
        })}
      />,
    )
    expect(screen.getByText(/text above is model-merged/)).toBeTruthy()
  })

  it("draws a bar chart without throwing", () => {
    const { container } = render(
      <ArtifactRenderer
        artifact={artifact({
          kind: "chart",
          type: "bar",
          encoding: {
            x: { key: "label", label: "Segment", kind: "categorical" },
            y: [{ key: "value", label: "Revenue", kind: "numeric", color_index: 1 }],
            series: null,
          },
        })}
      />,
    )
    expect(container.querySelector("figure")).toBeTruthy()
  })
})

describe("ChartBoundary", () => {
  it("falls back instead of unmounting the conversation", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {})
    function Broken(): never {
      throw new Error("recharts blew up")
    }
    render(
      <ChartBoundary fallback={<p>fallback table</p>}>
        <Broken />
      </ChartBoundary>,
    )
    expect(screen.getByText("fallback table")).toBeTruthy()
    spy.mockRestore()
  })
})
