import glob
import gzip
import io
import os
import re

import numpy as np
from scipy.io import hb_read, mmread
from scipy.sparse import coo_matrix, csc_array


def _edges_from_coo(matrix):
    """Undirected edge list (1-based, u < v, no loops, no duplicates)."""
    matrix = matrix.tocoo()
    rows = matrix.row.astype(np.int64) + 1
    cols = matrix.col.astype(np.int64) + 1
    mask = rows != cols
    u = np.minimum(rows[mask], cols[mask])
    v = np.maximum(rows[mask], cols[mask])
    pairs = np.unique(np.stack([u, v], axis=1), axis=0) if u.size else []
    return [(int(a), int(b)) for a, b in pairs]


def _open_text(path):
    if path.endswith(".gz"):
        return io.StringIO(gzip.decompress(open(path, "rb").read()).decode("ascii"))
    return open(path)


def _read_ints(lines, fmt, count):
    """Fixed-width ints as in an HB `(16I5)` format (fields may touch)."""
    width = int(re.search(r"I(\d+)", fmt, re.I).group(1))
    out = []
    for line in lines:
        line = line.rstrip("\r\n")
        for k in range(0, len(line), width):
            tok = line[k:k + width].strip()
            if tok:
                out.append(int(tok))
    return np.array(out[:count], dtype=np.int64)


def _parse_elemental(lines):
    """HB elemental (xxE) matrix, e.g. LAP 25 (PSE): rows are nodes and each
    column is a finite element listing its nodes. Assembled into the node-node
    pattern A @ A.T, i.e. every element becomes a clique."""
    ptrcrd, indcrd, _, rhscrd = (int(x) for x in lines[1].split()[1:5])
    nrow, ncol, nnz = (int(x) for x in lines[2][3:].split()[:3])
    ptrfmt, indfmt = lines[3][:16], lines[3][16:32]
    body = lines[4 + (1 if rhscrd > 0 else 0):]
    ptr = _read_ints(body[:ptrcrd], ptrfmt, ncol + 1) - 1
    ind = _read_ints(body[ptrcrd:ptrcrd + indcrd], indfmt, nnz) - 1
    a = csc_array((np.ones(len(ind), dtype=np.int32), ind, ptr),
                  shape=(nrow, ncol))
    return nrow, (a @ a.T)


def _parse_hb(path):
    lines = _open_text(path).read().splitlines()
    mxtype = lines[2][:3].upper()
    if mxtype[2] == "E":
        n, m = _parse_elemental(lines)
        return n, _edges_from_coo(m)
    m = hb_read(io.StringIO("\n".join(lines) + "\n"), spmatrix=False)
    return int(max(m.shape)), _edges_from_coo(m)


def _parse_mm(path):
    matrix = mmread(path, spmatrix=False)  # also reads .mtx.gz
    if not hasattr(matrix, "tocoo"):  # dense array format
        matrix = coo_matrix(matrix)
    return int(max(matrix.shape)), _edges_from_coo(matrix)


def parse_file(path):
    """Parse a Matrix Market (.mtx, .mtx.gz) or Harwell-Boeing (.xxx.gz)
    file as an undirected graph.

    Returns (dataset_name, vertex_count, edges): edges is a list of (u, v)
    tuples, 1-based (vertices are 1..vertex_count), u < v, no self-loops, no
    duplicates.
    """
    base = os.path.basename(path)
    if base.endswith(".gz"):
        base = base[:-3]
    stem, ext = os.path.splitext(base)
    name = re.sub(r"_+", "_", stem)  # NIST HB names look like lap___25

    if ext.lower() == ".mtx":
        n, edges = _parse_mm(path)
    else:
        n, edges = _parse_hb(path)
    return name, n, edges


parse_mtx = parse_file  # backwards-compatible name


def parse_all(root="dataset", skip_non_graphs=True, skipped=None):
    """Yield (dataset_name, vertex_count, edges) for every matrix file under
    `root`. Unparseable files are reported and skipped.

    With skip_non_graphs (default) these are also skipped, since they do not
    describe a graph:
      * non-square matrices, except elemental ones (HB type xxE), whose
        element-to-node structure is assembled into a node graph;
      * matrices that end up with no edges (e.g. diagonal mass matrices).
    If `skipped` is a list, (name, reason) tuples are appended to it so the
    report can cite them.
    """
    def skip(name, reason):
        print(f"SKIP {name}: {reason}")
        if skipped is not None:
            skipped.append((name, reason))

    paths = glob.glob(os.path.join(root, "**", "*.gz"), recursive=True)
    paths += glob.glob(os.path.join(root, "**", "*.mtx"), recursive=True)
    for path in sorted(paths):
        try:
            if skip_non_graphs:
                info = matrix_info(path)
                elemental = info["format"] == "HB" and info["type"][2:3] == "E"
                if not info["square"] and not elemental:
                    skip(info["name"], f"non-square {info['rows']}x{info['cols']}")
                    continue
            result = parse_file(path)
            if skip_non_graphs and not result[2]:
                skip(result[0], "no edges")
                continue
            yield result
        except Exception as e:
            skip(os.path.basename(path), f"{type(e).__name__}: {e}")


def matrix_info(path):
    """Read only the header of a matrix file (fast, no data parsing).

    Returns a dict: name, format ("MM" or "HB"), type (HB type code or MM
    qualifiers), rows, cols, entries, square.
    """
    name = os.path.basename(path)
    name = name[:-3] if name.endswith(".gz") else name
    stem, ext = os.path.splitext(name)
    with _open_text(path) as f:
        if ext.lower() == ".mtx":
            banner = f.readline().split()
            kind = " ".join(banner[3:]) if len(banner) > 3 else ""
            line = f.readline()
            while line.startswith("%"):
                line = f.readline()
            nums = line.split()
            fmt, rows, cols = "MM", int(nums[0]), int(nums[1])
            entries = int(nums[2]) if len(nums) > 2 else rows * cols
        else:
            f.readline(), f.readline()
            line = f.readline()
            fmt, kind = "HB", line[:3].upper()
            rows, cols, entries = (int(x) for x in line[3:].split()[:3])
    return {"name": re.sub(r"_+", "_", stem), "format": fmt, "type": kind,
            "rows": rows, "cols": cols, "entries": entries,
            "square": rows == cols}


def log_matrix_shapes(root="dataset", out_csv="matrix_shapes.csv"):
    """Write one CSV row per matrix under `root` (for the report) and return
    the list of non-square ones."""
    import csv

    paths = glob.glob(os.path.join(root, "**", "*.gz"), recursive=True)
    paths += glob.glob(os.path.join(root, "**", "*.mtx"), recursive=True)
    infos = []
    for path in sorted(paths):
        try:
            info = matrix_info(path)
        except Exception as e:
            print(f"SKIP {path}: {type(e).__name__}: {e}")
            continue
        info["set"] = os.path.basename(os.path.dirname(path))
        infos.append(info)
    fields = ["set", "name", "format", "type", "rows", "cols", "entries",
              "square"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(infos)
    return [i for i in infos if not i["square"]]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--shapes":
        root = sys.argv[2] if len(sys.argv) > 2 else "dataset"
        bad = log_matrix_shapes(root)
        for i in bad:
            print(f"{i['set']}/{i['name']}\t{i['type']}\t{i['rows']}x{i['cols']}")
        print(f"{len(bad)} non-square; full list in matrix_shapes.csv")
        sys.exit()

    target = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    if os.path.isdir(target):
        total = 0
        for name, n, e in parse_all(target):
            total += 1
            print(f"{name}\t{n}\t{len(e)}")
        print(f"parsed {total} files")
    else:
        name, n, e = parse_file(target)
        print(name, n, len(e))
