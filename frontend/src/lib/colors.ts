// The run palette — shared by the bar chart (bars) and the legend / table dots.
export const RUN_COLORS = ["#c6f24a", "#5ad1d6", "#f3b14a", "#b18cff", "#f1685e", "#7aa2ff"];

/** Color for run #i, cycling if there are more runs than colors. */
export const runColor = (i: number) => RUN_COLORS[i % RUN_COLORS.length];
