export const MODES = [
  { id: 'chat', label: 'Chat', icon: 'MessageSquare', hint: 'Ask anything about your paper...' },
  { id: 'draft', label: 'Draft', icon: 'PenLine', hint: 'Draft a section with AI assistance...' },
  { id: 'review', label: 'Review', icon: 'CheckCircle', hint: 'Review and improve existing text...' },
  { id: 'synthesize', label: 'Synthesize', icon: 'Layers', hint: 'Synthesize across multiple sources...' },
  { id: 'cite', label: 'Cite', icon: 'Quote', hint: 'Find and format citations...' },
  { id: 'transition', label: 'Transitions', icon: 'ArrowRightLeft', hint: 'Bridge between sections...' },
  { id: 'consistency', label: 'Consistency', icon: 'Shield', hint: 'Check terminology alignment...' },
  { id: 'structure', label: 'Structure', icon: 'LayoutList', hint: 'Analyze paper outline...' },
] as const;

export type ModeId = typeof MODES[number]['id'];

export const VISIBLE_MODES: ModeId[] = ['chat', 'draft', 'review', 'synthesize'];

export const MODEL_OPTIONS = [
  { value: '', label: 'Auto' },
  { value: 'openai/gpt-5.4', label: 'GPT-5.4' },
  { value: 'claude-opus-4-6', label: 'Opus' },
  { value: 'claude-sonnet-4-5-20250929', label: 'Sonnet' },
  { value: 'gemini-3.1-pro-preview', label: 'Gemini' },
] as const;

export const THINK_OPTIONS = [
  { value: '', label: 'Auto' },
  { value: '10000', label: 'On' },
  { value: '0', label: 'Off' },
] as const;

export const TEMPLATES = [
  { id: 'draft_section', label: 'Draft', mode: 'draft', requiresSection: true },
  { id: 'review_section', label: 'Review', mode: 'review', requiresSection: true },
  { id: 'consistency_check', label: 'Consistency', mode: 'consistency', requiresSection: true },
  { id: 'transition_bridge', label: 'Transition', mode: 'transition', requiresSection: true },
  { id: 'synth_topic', label: 'Synthesize', mode: 'synthesize', requiresSection: false },
  { id: 'full_audit', label: 'Full Audit', mode: 'structure', requiresSection: false },
] as const;
