import { describe, expect, it } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"

import {
  ErrorBanner,
  Markdown,
  ProvenanceBadge,
  SourcesPanel,
  SubQuestions,
} from "@/components/answer-details"
import type { QueryResponse, Source } from "@/lib/types"

const base: QueryResponse = {
  answer: "a",
  route: "HYBRID",
  original_query: "q",
  artifacts: [],
  segments: [],
}

describe("Markdown", () => {
  it("renders emphasis and lists instead of literal asterisks", () => {
    const { container } = render(<Markdown text={"**Data:** up\n\n- one\n- two"} />)
    expect(container.querySelector("strong")?.textContent).toBe("Data:")
    expect(container.querySelectorAll("li").length).toBe(2)
    expect(container.textContent).not.toContain("**")
  })

  it("does not render raw HTML", () => {
    const { container } = render(<Markdown text={'<img src=x onerror="alert(1)">hi'} />)
    expect(container.querySelector("img")).toBeNull()
  })
})

describe("ErrorBanner", () => {
  it("shows a failed leg that the prose would hide", () => {
    render(<ErrorBanner response={{ ...base, sql_error: "The query could not be run" }} />)
    expect(screen.getByText(/Table query failed/)).toBeTruthy()
    expect(screen.getByText(/uses what did succeed/)).toBeTruthy()
  })

  it("renders nothing when both legs worked", () => {
    const { container } = render(<ErrorBanner response={base} />)
    expect(container.textContent).toBe("")
  })
})

describe("ProvenanceBadge", () => {
  it("marks a merged answer as unverified", () => {
    render(<ProvenanceBadge segments={[{ kind: "unverified", text: "x" }]} />)
    expect(screen.getByText(/check figures against the chart/)).toBeTruthy()
  })
})

describe("SubQuestions", () => {
  it("lists each part with its verdict", () => {
    render(
      <SubQuestions
        items={[
          { question: "What was revenue?", verdict: "ANSWERED", source_ids: [1] },
          { question: "What was the target?", verdict: "NOT_FOUND", source_ids: [] },
        ]}
      />,
    )
    expect(screen.getByText("Covered")).toBeTruthy()
    expect(screen.getByText("Not in the documents")).toBeTruthy()
  })

  it("stays hidden for a single question", () => {
    const { container } = render(
      <SubQuestions items={[{ question: "q", verdict: "ANSWERED", source_ids: [] }]} />,
    )
    expect(container.textContent).toBe("")
  })
})

describe("SourcesPanel", () => {
  const sources: Source[] = [
    { score: 0.91, score_kind: "rerank", content: "Revenue rose 12%.", source_file: "report.pdf", page_number: 4 },
    { score: null, score_kind: "text", content: "Appendix.", source_file: "report.pdf" },
  ]

  it("is collapsed until opened, then shows file, page and snippet", () => {
    render(<SourcesPanel sources={sources} />)
    expect(screen.queryByText("Revenue rose 12%.")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: /2 sources/ }))
    expect(screen.getByText(/report\.pdf · p\. 4/)).toBeTruthy()
    expect(screen.getByText("Revenue rose 12%.")).toBeTruthy()
    expect(screen.getByText("relevance 0.91")).toBeTruthy()
    expect(screen.getByText("keyword match")).toBeTruthy()
  })
})
