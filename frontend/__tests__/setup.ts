import { afterEach } from "vitest"
import { cleanup } from "@testing-library/react"

// recharts' ResponsiveContainer measures its parent; jsdom has no layout.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver

afterEach(() => cleanup())
