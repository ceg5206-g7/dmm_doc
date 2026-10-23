# Project proposal

[Download the current proposal PDF](proposal.pdf).

The implementation contract uses **C++20**, matching `CMAKE_CXX_STANDARD 20`,
concepts and `std::span` in the public API. The approved language-standard
amendment updates both references in the proposal: the platform contract in
section 2 and the source/build deliverable in section 7. It changes the former
C/C++17 wording to C/C++20 without changing the experimental design or other text.

The six-page PDF was checked by comparing extracted text before/after (only
those two substitutions), checking the page count, and rendering both affected
pages for visual inspection. The original local PDF is preserved in the ignored
`results/proposal-original.pdf`; the ignored root `proposal.pdf` is synchronized
with this versioned documentation asset.

Other framework acceptance conditions remain in force. See
[framework milestones](framework.md) for actual implementation status; the PDF
is a proposal and does not itself establish that every task has been completed.
