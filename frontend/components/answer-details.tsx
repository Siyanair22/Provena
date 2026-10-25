"use client"

/**
 * Everything around an answer's prose that says where it came from.
 *
 * The backend has always returned citations (file, page, section, offsets),
 * leg errors and coverage verdicts; none of it reached the screen, so a user
 * could not tell a computed figure from a model's paraphrase or find the page
 * a sentence was drawn from. These components are that missing surface.
 */

import * as React from "react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  AlertTriangle,
  BookOpen,
  Calculator,
  ChevronDown,
  CircleCheck,
  CircleDashed,
  CircleSlash,
  FileText,
  Sparkles,
} from "lucide-react"

import type { AnswerSegment, QueryResponse, Source, SubQuestion } from "@/lib/types"

// --------------------------------------------------------------- markdown --

/**
 * Model answers are markdown: lists, bold, the occasional heading. Rendered as
 * a bare <p> the line breaks collapsed and "**Data Analysis:**" showed its
 * asterisks. react-markdown never renders raw HTML, so this stays text-safe.
 */
const MARKDOWN: Components = {
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="mb-2 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="pl-0.5">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  h1: ({ children }) => <p className="mb-1.5 mt-3 font-semibold first:mt-0">{children}</p>,
  h2: ({ children }) => <p className="mb-1.5 mt-3 font-semibold first:mt-0">{children}</p>,
  h3: ({ children }) => <p className="mb-1 mt-2 font-semibold first:mt-0">{children}</p>,
  code: ({ children }) => (
    <code className="rounded bg-secondary px-1 py-0.5 font-mono text-[0.85em]">{children}</code>
  ),
  pre: ({ children }) => (
    <pre className="mb-2 overflow-x-auto rounded-md bg-secondary p-2 text-xs">{children}</pre>
  ),
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer noopener" className="underline underline-offset-2">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="mb-2 overflow-x-auto">
      <table className="w-full text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => <th className="border-b px-2 py-1 text-left font-medium">{children}</th>,
  td: ({ children }) => <td className="border-b border-border/50 px-2 py-1">{children}</td>,
}

export function Markdown({ text }: { text: string }) {
  return (
    <div className="break-words text-sm leading-relaxed text-foreground">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={MARKDOWN}>
        {text}
      </ReactMarkdown>
    </div>
  )
}

// ------------------------------------------------------------- provenance --

const SEGMENT_STYLE: Record<
  AnswerSegment["kind"],
  { icon: typeof Calculator; label: string; className: string }
> = {
  computed: {
    icon: Calculator,
    label: "Written from table data",
    className: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
  },
  retrieved: {
    icon: BookOpen,
    label: "Written from cited passages",
    className: "border-sky-500/30 bg-sky-500/10 text-sky-400",
  },
  unverified: {
    icon: Sparkles,
    label: "Model-merged: check figures against the chart",
    className: "border-amber-500/30 bg-amber-500/10 text-amber-400",
  },
}

/** One badge for the answer's provenance. The chart carries its own. */
export function ProvenanceBadge({ segments }: { segments: AnswerSegment[] }) {
  const kinds = Array.from(new Set(segments.map((s) => s.kind)))
  if (kinds.length === 0) return null
  return (
    <div className="mb-2 flex flex-wrap gap-1.5">
      {kinds.map((kind) => {
        const style = SEGMENT_STYLE[kind]
        const Icon = style.icon
        const note = segments.find((s) => s.kind === kind)?.note
        return (
          <span
            key={kind}
            title={note}
            className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium ${style.className}`}
          >
            <Icon className="h-2.5 w-2.5" aria-hidden />
            {style.label}
          </span>
        )
      })}
    </div>
  )
}

// ----------------------------------------------------------------- errors --

/**
 * Failures the answer text alone would hide. On HYBRID a failed leg used to be
 * invisible: the prose simply came from the other one.
 */
export function ErrorBanner({ response }: { response: QueryResponse }) {
  const problems: string[] = []
  if (response.sql_error) problems.push(`Table query failed: ${response.sql_error}`)
  if (response.rag_error) problems.push(`Document search failed: ${response.rag_error}`)
  if (response.prose_error) problems.push(`Summary unavailable: ${response.prose_error}`)
  if (problems.length === 0) return null

  return (
    <div
      role="status"
      className="mb-2 flex items-start gap-1.5 rounded-md border border-amber-500/30 bg-amber-500/10 px-2.5 py-1.5 text-[11px] text-amber-300"
    >
      <AlertTriangle className="mt-px h-3 w-3 shrink-0" aria-hidden />
      <div className="space-y-0.5">
        {problems.map((p) => (
          <p key={p}>{p}</p>
        ))}
        {response.error ? null : (
          <p className="text-amber-300/70">The answer above uses what did succeed.</p>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------- sub-questions --

const VERDICT_STYLE = {
  ANSWERED: { icon: CircleCheck, label: "Covered", className: "text-emerald-400" },
  PARTIAL: { icon: CircleDashed, label: "Partly covered", className: "text-amber-400" },
  NOT_FOUND: { icon: CircleSlash, label: "Not in the documents", className: "text-red-400" },
} as const

export function SubQuestions({ items }: { items: SubQuestion[] }) {
  if (items.length < 2) return null
  return (
    <div className="mt-3 rounded-lg border border-[hsl(var(--glass-border))] px-3 py-2">
      <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        Question parts
      </p>
      <ul className="space-y-1">
        {items.map((item) => {
          const style = VERDICT_STYLE[item.verdict]
          const Icon = style.icon
          return (
            <li key={item.question} className="flex items-start gap-1.5 text-xs">
              <Icon className={`mt-0.5 h-3 w-3 shrink-0 ${style.className}`} aria-hidden />
              <span className="min-w-0 flex-1 text-foreground">{item.question}</span>
              <span className={`shrink-0 text-[10px] ${style.className}`}>
                {style.label}
                {item.source_ids.length > 0 && (
                  <span className="text-muted-foreground"> · {item.source_ids.map((i) => `[${i}]`).join(" ")}</span>
                )}
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

// ---------------------------------------------------------------- sources --

function scoreLabel(source: Source): string | null {
  if (source.score === null || source.score === undefined) return "keyword match"
  const kind = source.score_kind === "rerank" ? "relevance" : "similarity"
  return `${kind} ${source.score.toFixed(2)}`
}

function locator(source: Source): string {
  const parts = [source.source_file || "Unknown file"]
  if (source.page_number) parts.push(`p. ${source.page_number}`)
  return parts.join(" · ")
}

/**
 * Numbered to match the [Source n] labels the answer prompt uses, so a
 * citation in the prose can be looked up here.
 */
export function SourcesPanel({ sources, bridged = false }: { sources: Source[]; bridged?: boolean }) {
  const [open, setOpen] = React.useState(false)
  if (sources.length === 0) return null

  return (
    <div className="mt-3 rounded-lg border border-[hsl(var(--glass-border))]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-3 py-1.5 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        <span className="flex items-center gap-1.5">
          <FileText className="h-3 w-3" aria-hidden />
          {sources.length} {sources.length === 1 ? "source" : "sources"}
        </span>
        <ChevronDown className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
      </button>

      {open && (
        <ol className="space-y-2 border-t border-[hsl(var(--glass-border))] px-3 py-2">
          {sources.map((source, index) => (
            <li key={`${source.chunk_id}-${index}`} className="text-xs">
              <div className="flex flex-wrap items-baseline justify-between gap-x-2">
                <span className="font-medium text-foreground">
                  <span className="tabular-nums text-muted-foreground">[{index + 1}]</span>{" "}
                  {locator(source)}
                </span>
                <span className="text-[10px] tabular-nums text-muted-foreground">{scoreLabel(source)}</span>
              </div>
              {source.heading_context && (
                <p className="mt-0.5 truncate text-[10px] text-muted-foreground">{source.heading_context}</p>
              )}
              <p className="mt-1 whitespace-pre-wrap border-l-2 border-[hsl(var(--glass-border))] pl-2 text-[11px] leading-relaxed text-muted-foreground">
                {source.content}
              </p>
              {bridged && source.table_ids && source.table_ids.length > 0 && (
                <p className="mt-1 text-[10px] text-muted-foreground">
                  Table stub: the table itself was queried for this answer.
                </p>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
