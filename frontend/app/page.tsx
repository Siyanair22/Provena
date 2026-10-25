"use client"

import { useCallback, useState } from "react"
import { MessageSquare, Database, FileText, X, Zap } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DocumentSidebar } from "@/components/document-sidebar"
import { ChatInterface } from "@/components/chat-interface"
import { TablesView } from "@/components/tables-view"
import type { Document } from "@/lib/types"

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  // Shared by the sidebar (which sets it), the chat and the tables view. The
  // backend has always filtered on document_ids; nothing in the UI sent them.
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [documentCount, setDocumentCount] = useState(0)
  const handleDocuments = useCallback((docs: Document[]) => {
    const live = new Set(docs.filter((d) => d.status === "completed").map((d) => d.id))
    setDocumentCount(live.size)
    // Drop selections for documents that no longer exist (the backend wipes
    // everything after inactivity), or scoping would silently match nothing.
    setSelectedIds((prev) => {
      const kept = prev.filter((id) => live.has(id))
      return kept.length === prev.length ? prev : kept
    })
  }, [])

  return (
    <div className="relative flex h-dvh overflow-hidden bg-background">
      {/* Background gradient orbs */}
      <div className="pointer-events-none fixed inset-0" aria-hidden="true">
        <div className="absolute -left-32 -top-32 h-96 w-96 rounded-full bg-[hsl(224,76%,56%)]/[0.04] blur-3xl" />
        <div className="absolute -bottom-32 -right-32 h-96 w-96 rounded-full bg-[hsl(270,60%,58%)]/[0.04] blur-3xl" />
      </div>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-background/60 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
          onKeyDown={(e) => {
            if (e.key === "Escape") setSidebarOpen(false)
          }}
          role="button"
          tabIndex={0}
          aria-label="Close sidebar overlay"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`glass-strong fixed z-40 flex h-full w-80 flex-col border-r border-[hsl(225,12%,18%)] transition-transform duration-300 lg:relative lg:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Sidebar mobile close */}
        <div className="flex items-center justify-between border-b border-[hsl(225,12%,18%)] px-4 py-3 lg:hidden">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-[hsl(224,76%,56%)]" />
            <span className="text-sm font-semibold text-foreground">
              Provena
            </span>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="rounded-md p-1.5 text-muted-foreground hover:bg-secondary hover:text-foreground"
            aria-label="Close sidebar"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Desktop sidebar logo */}
        <div className="hidden items-center gap-2 border-b border-[hsl(225,12%,18%)] px-4 py-3 lg:flex">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[hsl(224,76%,56%)]/20">
            <Zap className="h-3.5 w-3.5 text-[hsl(224,76%,56%)]" />
          </div>
          <span className="text-sm font-bold text-foreground">Provena</span>
        </div>

        <DocumentSidebar
          selectedIds={selectedIds}
          onSelectionChange={setSelectedIds}
          onDocumentsChange={handleDocuments}
        />
      </aside>

      {/* Main content */}
      <main className="relative flex min-w-0 flex-1 flex-col">
        <Tabs defaultValue="chat" className="flex h-full flex-col">
          {/* Top bar */}
          <div className="glass-strong z-10 flex items-center gap-2 border-b border-[hsl(225,12%,18%)] px-4 py-2">
            {/* Mobile sidebar toggle */}
            <button
              onClick={() => setSidebarOpen(true)}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground lg:hidden"
              aria-label="Open sidebar"
            >
              <FileText className="h-4 w-4" />
            </button>

            <TabsList className="h-9 bg-secondary/50">
              <TabsTrigger
                value="chat"
                className="gap-1.5 text-xs data-[state=active]:bg-[hsl(225,15%,12%)] data-[state=active]:text-foreground"
              >
                <MessageSquare className="h-3.5 w-3.5" />
                <span>Chat</span>
              </TabsTrigger>
              <TabsTrigger
                value="tables"
                className="gap-1.5 text-xs data-[state=active]:bg-[hsl(225,15%,12%)] data-[state=active]:text-foreground"
              >
                <Database className="h-3.5 w-3.5" />
                <span>Tables</span>
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="chat" className="mt-0 flex-1 overflow-hidden">
            <ChatInterface documentIds={selectedIds} documentCount={documentCount} />
          </TabsContent>

          <TabsContent value="tables" className="mt-0 flex-1 overflow-hidden">
            <TablesView documentIds={selectedIds} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
