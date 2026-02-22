# Design System — MBA Paper Agent

## Direction
Personality: Academic Library — Warmth & Sophistication
Foundation: Warm (cream/parchment)
Depth: Borders only (no shadows except input focus)
Target: Academic writing tool — MBA researcher audience

## Tokens

### Spacing (4px base)
Scale: 4, 8, 12, 16, 20, 24, 28, 32, 48
- Micro gap: 4px
- Element gap: 8px
- Card padding: 12px (compact) / 16px (standard)
- Section gap: 24px
- Message gap: 28px
- Page padding: 20px

### Colors
--bg:               #FAF8F4 (warm cream — page background)
--bg-sidebar:       #F3EDE4 (warm beige — sidebar)
--bg-sidebar-hover: #EAE3D8
--bg-sidebar-active:#E2DACF
--bg-card:          #FFFFFF (white — cards, inputs)
--bg-paper:         #FDFCFA (near-white — doc panel)
--bg-user:          #F0EBE2 (warm — user messages)
--bg-code:          #F5F0E8 (warm — code blocks, thinking)
--bg-dark:          #2D2A26 (charcoal — toast)
--border:           #DDD6CB (default borders)
--border-light:     #EAE3D8 (subtle dividers)
--border-focus:     #B5A08A (focus, active states)
--text:             #2D2A26 (near-black)
--text-secondary:   #6B6560
--text-muted:       #9A948C
--text-inverse:     #FDFCFA
--accent:           #8B6E4E (warm bronze — primary accent)
--accent-hover:     #7A5F42
--accent-soft:      #D4C4AE (selections, highlights)
--green:            #5A8C6F (success, progress)
--red:              #B85C5C (error, delete)
--blue:             #5A7EB8 (info, Sonnet badge)

### Typography
UI: Inter (400, 500, 600)
Content: Lora (400, 500, 600, italic) — serif for agent responses + doc viewer
Mono: IBM Plex Mono (400, 500) — code, stats, badges
- Empty state title: 24px / Lora 500
- Sidebar brand: 15px / Lora 600
- Body: 14px / Inter 400
- Agent content: 15px / Lora 400
- Mode tag: 10px / Inter 500 / uppercase
- Stat row: 11px / IBM Plex Mono 400

### Radius
- Small (buttons, cards, inputs): 6px
- Medium (input box, user bubble): 10px
- Large (doc panel border-radius): 16px
- Pill (mode pills, overrides, toast): 24px

## Patterns

### Sidebar
- Background: var(--bg-sidebar)
- Border-right: 1px solid var(--border-light)
- Fixed width: 260px
- Session items: 8px 10px padding, 6px radius

### Mode Pills
- Container: var(--bg-sidebar), pill radius, 3px padding
- Inactive: transparent, muted text
- Active: white bg, shadow, primary text
- 11.5px font, 5px 12px padding

### Message (User)
- Background: var(--bg-user)
- Radius: 10px
- Padding: 12px 16px
- Font: Inter 14px

### Message (Agent)
- No background
- Font: Lora 15px
- Headings use Inter (switch to sans-serif for structure)

### Thinking Box
- Left border: 3px solid var(--accent-soft)
- Background: var(--bg-code)
- Collapsible via toggle button
- Mono font, 12px

### Input Box
- White background, 1px border
- Radius: 10px
- Focus: border-focus + subtle shadow
- Send button: 32px circle, dark bg, accent on hover

### Override Pills
- 10px mono font
- Pill radius, 2px 8px padding
- Active: white bg, focus border
- Opus active: accent color
- Sonnet active: blue color

## Class Naming Convention
Readable BEM-like names (not full BEM):
- Layout: .sidebar, .main, .toolbar, .chat-area, .input-area, .doc-panel
- Components: .mode-pill, .session-item, .message, .thinking-box
- States: .active, .is-open, .is-visible, .collapsed, .has-docs
- Elements: .msg-label, .msg-content, .msg-footer, .source-chip
- Modifiers: .is-user, .is-agent, .user-content, .agent-content
