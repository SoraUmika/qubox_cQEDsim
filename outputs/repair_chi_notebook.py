import json
from pathlib import Path


NOTEBOOK = Path(r"c:\Users\dazzl\Box\Shyam Shankar Quantum Circuits Group\Users\Users_JianJun\cQED_simulation\chi_evolution_copy.ipynb")


def normalize(lines):
    return [line.rstrip("\n") for line in lines]


def replace_source(cell, old_lines, new_lines):
    source = cell.get("source", [])
    normalized_source = normalize(source)
    normalized_old = normalize(old_lines)
    for idx in range(len(source) - len(old_lines) + 1):
        if normalized_source[idx:idx + len(old_lines)] == normalized_old:
            replacement = [line if line.endswith("\n") else f"{line}\n" for line in new_lines]
            if source[idx + len(old_lines):] and not source[idx + len(old_lines)].endswith("\n"):
                replacement[-1] = replacement[-1].rstrip("\n")
            cell["source"] = source[:idx] + replacement + source[idx + len(old_lines):]
            return True
    return False


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
    cells = notebook["cells"]

    source_cell = None
    drive_cell = None
    for cell in cells:
        source = cell.get("source", [])
        if any("def fock_resolved_bloch_contributions(states, n_levels):" in line for line in source):
            source_cell = cell
        if any("Gaussian drive area target = pi rotation" in line for line in source):
            drive_cell = cell

    if source_cell is None or drive_cell is None:
        raise RuntimeError("Could not locate expected notebook cells to repair.")

    replacements = [
        (
            source_cell,
            ["                y_by_n[n, ti] = p_n * 2.0 * float(np.imag(rho_q_n[0, 1]))"],
            ["                y_by_n[n, ti] = p_n * float(np.real((rho_q_n * qt.sigmay()).tr()))"],
        ),
        (
            source_cell,
            [
                "def y_convention_pair(state):",
                "    rho_q = qt.ptrace(state, 0)",
                "    y_qubox = 2.0 * float(np.imag(rho_q[0, 1]))",
                "    y_sigma_y = float(np.real((rho_q * qt.sigmay()).tr()))",
                "    return y_qubox, y_sigma_y",
            ],
            [
                "def standard_and_legacy_y_components(state):",
                "    rho_q = qt.ptrace(state, 0)",
                "    y_sigma_y = float(np.real((rho_q * qt.sigmay()).tr()))",
                "    y_legacy = 2.0 * float(np.imag(rho_q[0, 1]))",
                "    return y_sigma_y, y_legacy",
            ],
        ),
        (
            source_cell,
            ["axes[0].plot(tlist * 1e6, y_sup, label=\"Y (qubox)\", color=\"C1\")"],
            ["axes[0].plot(tlist * 1e6, y_sup, label=\"Y\", color=\"C1\")"],
        ),
        (
            source_cell,
            ["    r\"Fock-resolved $Y_n = P(n) Y_{q|n}$ (qubox)\","],
            ["    r\"Fock-resolved $Y_n = P(n) Y_{q|n}$\","],
        ),
        (
            source_cell,
            ["y_qubox_init, y_sigma_y_init = y_convention_pair(psi_init_free_sup)"],
            ["y_sigma_y_init, y_legacy_init = standard_and_legacy_y_components(psi_init_free_sup)"],
        ),
        (
            source_cell,
            [
                "print(f\"Initial Y in qubox convention = {y_qubox_init:.6f}\")",
                "print(f\"Standard <sigma_y> = {y_sigma_y_init:.6f} = -Y_qubox\")",
                "print(\"The notebook plots use the same Y convention as qubox: Y = 2 Im(rho_ge).\")",
            ],
            [
                "print(f\"Initial Y = <sigma_y> = {y_sigma_y_init:.6f}\")",
                "print(f\"Legacy qubox-style Y_legacy = {y_legacy_init:.6f} = -Y\")",
                "print(\"All Bloch traces in this notebook use the standard Pauli convention Y = <sigma_y>.\")",
            ],
        ),
        (
            drive_cell,
            ["plt.plot(tlist_drive * 1e9, y_drive, label=\"Y (qubox)\")"],
            ["plt.plot(tlist_drive * 1e9, y_drive, label=\"Y\")"],
        ),
        (
            drive_cell,
            ["ax.text(0, 1.08, 0, \"Yq\")"],
            ["ax.text(0, 1.08, 0, \"Y\")"],
        ),
        (
            drive_cell,
            ["ax.set_ylabel(\"Y (qubox)\")"],
            ["ax.set_ylabel(\"Y\")"],
        ),
        (
            drive_cell,
            ["print(f\"Final Bloch vector = ({x_drive[-1]:.4f}, {y_drive[-1]:.4f}, {z_drive[-1]:.4f})\")"],
            [
                "print(f\"Final Bloch vector = ({x_drive[-1]:.4f}, {y_drive[-1]:.4f}, {z_drive[-1]:.4f})\")",
                "print(\"This notebook reports the Bloch vector using the standard Pauli convention Y = <sigma_y>.\")",
            ],
        ),
    ]

    failed = []
    for cell, old_lines, new_lines in replacements:
        if not replace_source(cell, old_lines, new_lines):
            failed.append(old_lines[0])

    if failed:
        raise RuntimeError("Failed replacements: " + "; ".join(failed))

    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8-sig")


if __name__ == "__main__":
    main()