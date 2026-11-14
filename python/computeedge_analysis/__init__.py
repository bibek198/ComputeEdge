"""ComputeEdge analysis: loading measurement sessions, checking claims, reports and paper figures.

Layers (no module imports from a layer above it):
  paths, records, units, stats   -- foundations
  io                             -- reads the immutable evidence in data/ into typed records
  style                          -- chart styling (presentation)
  claims                         -- pre-registered claims evaluated per session
  reports, figures               -- presentation of results
  cli                            -- command-line entry points (ce-claims, ce-report, ce-figures)
"""
