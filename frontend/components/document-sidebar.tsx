"use client"

import React from "react"

import { useCallback, useEffect, useRef, useState } from "react"
import {
  FileText,
  Upload,
  CheckCircle2,
  Clock,
  XCircle,
  Database,
  Layers,
  RefreshCw,
  Loader2,
  BookOpen,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { Document, UploadStatus } from "@/lib/types"
import { fetchDocuments, uploadDocument } from "@/lib/api"

interface DocumentSidebarProps {
  /** Documents the chat is scoped to. Empty means all. */
  selectedIds: string[]
  onSelectionChange: (ids: string[]) => void
  onDocumentsChange?: (documents: Document[]) => void
}

export function DocumentSidebar({
  selectedIds,
  onSelectionChange,
  onDocumentsChange,
}: DocumentSidebarProps) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle")
  const [uploadMessage, setUploadMessage] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const loadDocuments = useCallback(
    () =>
      fetchDocuments()
        .then((data) => {
          const docs = Array.isArray(data) ? data : []
          setDocuments(docs)
          onDocumentsChange?.(docs)
        })
        .catch(() => {
          // Silently fail on auto-refresh
        }),
    [onDocumentsChange],
  )

  const toggleSelected = (id: string) => {
    onSelectionChange(
      selectedIds.includes(id) ? selectedIds.filter((x) => x !== id) : [...selectedIds, id],
    )
  }

  const refreshDocuments = useCallback(async () => {
    setIsRefreshing(true)
    await loadDocuments()
    setIsRefreshing(false)
  }, [loadDocuments])

  useEffect(() => {
    loadDocuments()
    intervalRef.current = setInterval(loadDocuments, 12000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [loadDocuments])

  const validateFile = (file: File): string | null => {
    if (file.type !== "application/pdf") {
      return "Only PDF files are allowed"
    }
    if (file.size > 50 * 1024 * 1024) {
      return "File size must be under 50MB"
    }
    return null
  }

  const resetUploadLater = (ms: number) => {
    setTimeout(() => {
      setUploadStatus("idle")
      setUploadMessage(null)
      setUploadError(null)
    }, ms)
  }

  const handleUpload = async (file: File) => {
    const error = validateFile(file)
    if (error) {
      setUploadError(error)
      setUploadStatus("error")
      resetUploadLater(3000)
      return
    }

    setUploadStatus("uploading")
    setUploadMessage(null)
    setUploadError(null)

    // The request blocks while the backend OCRs, parses and embeds, and it
    // reports no progress. The old bar invented one that stalled at 90%;
    // an honest indeterminate state is shown instead, and the list below
    // polls the real per-stage status.
    const poll = setInterval(loadDocuments, 3000)

    try {
      const result = await uploadDocument(file)
      if (result.skipped) {
        setUploadStatus("skipped")
        setUploadMessage("Already uploaded; nothing to do")
      } else {
        setUploadStatus("success")
        setUploadMessage(
          `${result.table_count} ${result.table_count === 1 ? "table" : "tables"}, ` +
            `${result.chunk_count} text chunks`,
        )
      }
      resetUploadLater(4000)
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed")
      setUploadStatus("error")
      resetUploadLater(6000)
    } finally {
      clearInterval(poll)
      await loadDocuments()
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleUpload(file)
    e.target.value = ""
  }

  const statusConfig: Record<string, { icon: typeof CheckCircle2; color: string; label: string }> = {
    completed: {
      icon: CheckCircle2,
      color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
      label: "Completed",
    },
    pending: {
      icon: Clock,
      color: "bg-amber-500/20 text-amber-400 border-amber-500/30",
      label: "Pending",
    },
    extracting: {
      icon: Clock,
      color: "bg-amber-500/20 text-amber-400 border-amber-500/30",
      label: "Extracting",
    },
    parsing_tables: {
      icon: Clock,
      color: "bg-amber-500/20 text-amber-400 border-amber-500/30",
      label: "Parsing Tables",
    },
    chunking: {
      icon: Clock,
      color: "bg-amber-500/20 text-amber-400 border-amber-500/30",
      label: "Chunking",
    },
    failed: {
      icon: XCircle,
      color: "bg-red-500/20 text-red-400 border-red-500/30",
      label: "Failed",
    },
  }

  return (
    <aside className="flex h-full w-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 pb-3">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[hsl(224,76%,56%)]/20">
            <FileText className="h-4 w-4 text-[hsl(224,76%,56%)]" />
          </div>
          <h2 className="text-sm font-semibold text-foreground">Documents</h2>
        </div>
        <button
          onClick={refreshDocuments}
          className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          aria-label="Refresh documents"
        >
          <RefreshCw
            className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`}
          />
        </button>
      </div>

      {/* Upload Area */}
      <div className="px-4 pb-3">
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() =>
            uploadStatus !== "uploading" && fileInputRef.current?.click()
          }
          className={`glass-subtle group relative cursor-pointer rounded-xl p-4 transition-all ${
            isDragging
              ? "border-[hsl(224,76%,56%)] bg-[hsl(224,76%,56%)]/10"
              : uploadStatus === "error"
                ? "border-red-500/40 bg-red-500/5"
                : uploadStatus === "success"
                  ? "border-emerald-500/40 bg-emerald-500/5"
                  : "hover:border-[hsl(224,76%,56%)]/40 hover:bg-[hsl(225,15%,12%)]/50"
          } ${uploadStatus === "uploading" ? "pointer-events-none" : ""}`}
          role="button"
          tabIndex={0}
          aria-disabled={uploadStatus === "uploading"}
          onKeyDown={(e) => {
            if ((e.key === "Enter" || e.key === " ") && uploadStatus !== "uploading") {
              e.preventDefault()
              fileInputRef.current?.click()
            }
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileSelect}
            className="hidden"
            aria-label="Upload PDF file"
          />

          <div className="flex flex-col items-center gap-2 text-center">
            <div
              className={`rounded-full p-2 transition-colors ${
                isDragging
                  ? "bg-[hsl(224,76%,56%)]/20"
                  : "bg-secondary group-hover:bg-[hsl(224,76%,56%)]/10"
              }`}
            >
              <Upload
                className={`h-4 w-4 ${
                  isDragging
                    ? "text-[hsl(224,76%,56%)]"
                    : "text-muted-foreground group-hover:text-[hsl(224,76%,56%)]"
                }`}
              />
            </div>
            <div>
              <p className="flex items-center justify-center gap-1.5 text-xs font-medium text-foreground">
                {uploadStatus === "uploading" && <Loader2 className="h-3 w-3 animate-spin" aria-hidden />}
                {uploadStatus === "uploading"
                  ? "Processing…"
                  : uploadStatus === "success"
                    ? "Document ready"
                    : uploadStatus === "skipped"
                      ? "Already uploaded"
                      : uploadStatus === "error"
                        ? "Upload failed"
                        : "Drop PDF here"}
              </p>
              <p
                className={`mt-0.5 text-[10px] ${uploadError ? "text-red-400" : "text-muted-foreground"}`}
                role={uploadError ? "alert" : undefined}
              >
                {uploadError ||
                  uploadMessage ||
                  (uploadStatus === "uploading"
                    ? "OCR, table extraction and embedding can take a minute"
                    : "PDF files up to 50MB")}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Document List */}
      <div className="flex items-center gap-2 px-4 pb-2">
        <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          Uploaded Files
        </span>
        <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-secondary px-1 text-[10px] font-medium text-muted-foreground">
          {documents.length}
        </span>
        {selectedIds.length > 0 && (
          <button
            type="button"
            onClick={() => onSelectionChange([])}
            className="ml-auto text-[10px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
          >
            Use all
          </button>
        )}
      </div>
      {documents.some((d) => d.status === "completed") && (
        <p className="px-4 pb-2 text-[10px] text-muted-foreground">
          {selectedIds.length > 0
            ? `Chat answers from ${selectedIds.length} selected`
            : "Tick documents to limit answers to them"}
        </p>
      )}

      <ScrollArea className="flex-1 px-4 pb-4">
        <div className="flex flex-col gap-2">
          {documents.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-8 text-center">
              <FileText className="h-8 w-8 text-muted-foreground/40" />
              <p className="text-xs text-muted-foreground">
                No documents uploaded yet
              </p>
            </div>
          ) : (
            documents.map((doc) => {
              const status = statusConfig[doc.status] || statusConfig.pending
              const StatusIcon = status.icon
              const selectable = doc.status === "completed"
              const selected = selectedIds.includes(doc.id)
              return (
                <div
                  key={doc.id}
                  className={`glass-subtle group rounded-lg p-3 transition-colors hover:bg-[hsl(225,15%,14%)] ${
                    selected ? "ring-1 ring-[hsl(224,76%,56%)]/60" : ""
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex min-w-0 flex-1 items-start gap-2">
                      {selectable ? (
                        <input
                          type="checkbox"
                          checked={selected}
                          onChange={() => toggleSelected(doc.id)}
                          aria-label={`Limit answers to ${doc.filename}`}
                          className="mt-0.5 h-3.5 w-3.5 shrink-0 cursor-pointer accent-[hsl(224,76%,56%)]"
                        />
                      ) : (
                        <FileText className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                      )}
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-xs font-medium text-foreground" title={doc.filename}>
                          {doc.filename}
                        </p>
                        <div className="mt-1 flex items-center gap-2">
                          <Badge
                            className={`${status.color} border px-1.5 py-0 text-[10px]`}
                          >
                            <StatusIcon className="mr-1 h-2.5 w-2.5" />
                            {status.label}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>

                  {doc.status === "completed" && (
                    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 pl-5">
                      {doc.page_count != null && (
                        <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                          <BookOpen className="h-2.5 w-2.5" />
                          {doc.page_count} {doc.page_count === 1 ? "page" : "pages"}
                        </span>
                      )}
                      <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                        <Database className="h-2.5 w-2.5" />
                        {doc.table_count} {doc.table_count === 1 ? "table" : "tables"}
                      </span>
                      <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                        <Layers className="h-2.5 w-2.5" />
                        {doc.chunk_count} chunks
                      </span>
                    </div>
                  )}

                  {doc.status === "failed" && doc.error_message && (
                    <p className="mt-2 line-clamp-3 pl-5 text-[10px] leading-snug text-red-400" title={doc.error_message}>
                      {doc.error_message}
                    </p>
                  )}
                </div>
              )
            })
          )}
        </div>
      </ScrollArea>
    </aside>
  )
}
