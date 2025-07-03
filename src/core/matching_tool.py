import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union


# 1) 매칭 전처리
def indexing(poly1, poly2):
    # 고유 인덱스, 면적, spatial join
    poly1['poly1_idx'] = range(1, len(poly1) + 1)
    poly1 = poly1.reset_index(drop=True)

    poly2['poly2_idx'] = range(1, len(poly2) + 1)
    poly2 = poly2.reset_index(drop=True)

    poly1_area = poly1.geometry.area
    poly2_area = poly2.geometry.area

    poly1 = poly1.drop(columns=['area'], errors='ignore')
    idx_loc1 = poly1.columns.get_loc('poly1_idx')
    poly1.insert(loc=idx_loc1, column='area', value=poly1_area)

    poly2 = poly2.drop(columns=['area'], errors='ignore')
    idx_loc2 = poly2.columns.get_loc('poly2_idx')
    poly2.insert(loc=idx_loc2, column='area', value=poly2_area)

    outer_joined = outer_join(poly1, poly2, poly1_prefix="poly1", poly2_prefix="poly2")
    return poly1, poly2, outer_joined


def outer_join(poly1, poly2, poly1_prefix="poly1", poly2_prefix="poly2"):
    left_join = gpd.sjoin(poly1, poly2, how='left', predicate='intersects')
    right_join = gpd.sjoin(poly2, poly1, how='left', predicate='intersects')
    left_join.columns = [
        col.replace('_left', '_poly1').replace('_right', '_poly2')
        for col in left_join.columns
    ]

    right_join.columns = [
        col.replace('_left', '_poly2').replace('_right', '_poly1')
        for col in right_join.columns
    ]

    joined = pd.merge(left_join, right_join, how='outer', on=list(set(left_join.columns) & set(right_join.columns)))
    subset_cols = [f"{poly1_prefix}_idx", f"{poly2_prefix}_idx"]
    joined = joined.drop_duplicates(subset=subset_cols)
    joined = joined.reset_index(drop=True)
    return joined


# 2) 그래프 구성 및 정제
def build_graph(joined_df):
    # 공간적으로 매칭된 poly1, poly2 쌍으로부터 노드-링크 구조의 관계 그래프 생성
    nodes = set()
    links = []

    for _, row in joined_df.dropna(subset=["poly1_idx", "poly2_idx"]).iterrows():
        p1 = f"p1_{int(row['poly1_idx'])}"
        p2 = f"p2_{int(row['poly2_idx'])}"
        links.append({"source": p1, "target": p2})
        nodes.update([p1, p2])

    if "poly1_idx" in joined_df.columns:
        for p1 in joined_df["poly1_idx"].dropna().unique():
            nodes.add(f"p1_{int(p1)}")
    if "poly2_idx" in joined_df.columns:
        for p2 in joined_df["poly2_idx"].dropna().unique():
            nodes.add(f"p2_{int(p2)}")

    node_list = [{"id": n} for n in sorted(nodes)]

    return {"nodes": node_list, "links": links}


def add_energy_to_links(poly1, poly2, graph_dict):
    # 각 링크에 대해 IoU 계산
    poly1 = poly1.set_index("poly1_idx")
    poly2 = poly2.set_index("poly2_idx")

    for link in graph_dict["links"]:
        p1_idx = int(link["source"].replace("p1_", ""))
        p2_idx = int(link["target"].replace("p2_", ""))

        if p1_idx not in poly1.index or p2_idx not in poly2.index:
            raise ValueError(f"Missing poly1_idx {p1_idx} or poly2_idx {p2_idx} in geometry.")

        geom1 = poly1.loc[p1_idx, "geometry"]
        geom2 = poly2.loc[p2_idx, "geometry"]

        if geom1.is_empty or geom2.is_empty:
            raise ValueError(f"Empty geometry at poly1_idx {p1_idx} or poly2_idx {p2_idx}.")

        intersection = geom1.intersection(geom2)
        union = geom1.union(geom2)

        if union.area == 0 or intersection.area == 0:
            raise ValueError(f"No valid overlap between {p1_idx} and {p2_idx}.")

        link["energy"] = intersection.area / union.area

    return graph_dict


def split_graph_by_energy(poly1, poly2, graph_dict, threshold):
    # energy(IoU)가 낮은 링크 끊고 component 및 link 재구성
    poly1 = poly1.set_index("poly1_idx")
    poly2 = poly2.set_index("poly2_idx")

    G = nx.Graph()
    original_links = graph_dict["links"]
    original_nodes = graph_dict["nodes"]

    cut_links = []
    kept_links = []

    suppression = 0.7

    for link in original_links:
        energy = link.get("energy", 0)

        if energy >= threshold:
            G.add_edge(link["source"], link["target"], energy=energy)
            kept_links.append(link)
        else:
            p1_idx = int(link["source"].replace("p1_", ""))
            p2_idx = int(link["target"].replace("p2_", ""))
            geom1 = poly1.loc[p1_idx, "geometry"]
            geom2 = poly2.loc[p2_idx, "geometry"]
            area1 = poly1.loc[p1_idx, "area"]

            intersection = geom1.intersection(geom2)
            ol1 = intersection.area / area1 if area1 > 0 else 0

            if ol1 < suppression:
                cut_links.append(link)
            else:
                G.add_edge(link["source"], link["target"], energy=energy)
                kept_links.append(link)

    for node in original_nodes:
        G.add_node(node["id"])

    components_dict = {}
    new_node_list = []
    new_link_list = []

    for comp_idx, comp in enumerate(nx.connected_components(G)):
        poly1_set = sorted(int(n[3:]) for n in comp if n.startswith("p1_"))
        poly2_set = sorted(int(n[3:]) for n in comp if n.startswith("p2_"))

        components_dict[comp_idx] = {
            "poly1_set": poly1_set,
            "poly2_set": poly2_set
        }

        for n in comp:
            new_node_list.append({"id": n, "comp_idx": comp_idx})

        for u, v in G.subgraph(comp).edges:
            source, target = (u, v) if u.startswith("p1_") else (v, u)
            new_link_list.append({
                "source": source,
                "target": target,
                "comp_idx": comp_idx,
                "energy": G[source][target]["energy"]
            })

    summary = {
        "after_components": len(components_dict),
        "num_cut_links": len(cut_links)
    }
    return components_dict, {"nodes": new_node_list, "links": new_link_list}, cut_links, summary


# 3) 그래프 기반 정량 지표 계산
def mark_cut_links(poly1, poly2, cut_links):
    # link 제거 정보 기록
    cut_poly1_idxs = set(
        int(link["source"].replace("p1_", "")) for link in cut_links if link["source"].startswith("p1_"))
    cut_poly2_idxs = set(
        int(link["target"].replace("p2_", "")) for link in cut_links if link["target"].startswith("p2_"))

    poly1 = poly1.copy()
    poly2 = poly2.copy()

    poly1["cut_link"] = poly1["poly1_idx"].isin(cut_poly1_idxs)
    poly2["cut_link"] = poly2["poly2_idx"].isin(cut_poly2_idxs)

    return poly1, poly2


def attach_metrics_from_components(components_dict, poly1, poly2):
    # Relation, IoU, Overlap1,2 유형별 기록
    poly1 = poly1.copy()
    poly2 = poly2.copy()

    # 초기화
    metric_cols = [
        "comp_idx", "Relation",
        "iou_1n", "ol_pl1_1n", "ol_pl2_1n",
        "iou_n1", "ol_pl1_n1", "ol_pl2_n1",
        "iou_11", "ol_pl1_11", "ol_pl2_11",
        "iou_nn", "ol_pl1_nn", "ol_pl2_nn"
    ]
    for col in metric_cols[2:]:
        poly1[col] = np.nan
        poly2[col] = np.nan
    poly1["comp_idx"] = np.nan
    poly2["comp_idx"] = np.nan
    poly1["Relation"] = np.nan
    poly2["Relation"] = np.nan

    def calc_metrics(g1, g2):
        if g1 is None or g2 is None:
            return (np.nan, np.nan, np.nan)
        inter = g1.intersection(g2).area
        if inter == 0:
            return (0, 0, 0)
        return (
            inter / g1.union(g2).area,
            inter / g1.area if g1.area > 0 else 0,
            inter / g2.area if g2.area > 0 else 0
        )

    for comp_idx, comp in components_dict.items():
        p1_set = comp["poly1_set"]
        p2_set = comp["poly2_set"]

        rel = (
            "1:0" if len(p2_set) == 0 else
            "0:1" if len(p1_set) == 0 else
            "1:1" if len(p1_set) == 1 and len(p2_set) == 1 else
            "1:N" if len(p1_set) == 1 else
            "N:1" if len(p2_set) == 1 else "N:N"
        )

        poly1["Relation"] = poly1["Relation"].astype("object")
        poly2["Relation"] = poly2["Relation"].astype("object")

        poly1.loc[poly1['poly1_idx'].isin(p1_set), ["comp_idx", "Relation"]] = [comp_idx, rel]
        poly2.loc[poly2['poly2_idx'].isin(p2_set), ["comp_idx", "Relation"]] = [comp_idx, rel]

        if rel == "1:N":
            g1 = poly1.loc[poly1['poly1_idx'] == p1_set[0], "geometry"].values[0]
            g2_union = unary_union(poly2.loc[poly2['poly2_idx'].isin(p2_set), "geometry"])

            # n1: poly1 vs union(poly2)
            iou_n1, ol1_n1, ol2_n1 = calc_metrics(g1, g2_union)
            poly1.loc[poly1['poly1_idx'] == p1_set[0], ["iou_n1", "ol_pl1_n1", "ol_pl2_n1"]] = [iou_n1, ol1_n1, ol2_n1]
            poly2.loc[poly2['poly2_idx'].isin(p2_set), ["iou_n1", "ol_pl1_n1", "ol_pl2_n1"]] = [iou_n1, ol1_n1, ol2_n1]

            # nn: poly1 vs each poly2
            for p2 in p2_set:
                g2 = poly2.loc[poly2['poly2_idx'] == p2, "geometry"].values[0]
                iou, ol1, ol2 = calc_metrics(g1, g2)
                poly2.loc[poly2['poly2_idx'] == p2, ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]
            poly1.loc[poly1['poly1_idx'] == p1_set[0], ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]

        elif rel == "N:1":
            g2 = poly2.loc[poly2['poly2_idx'] == p2_set[0], "geometry"].values[0]
            g1_union = unary_union(poly1.loc[poly1['poly1_idx'].isin(p1_set), "geometry"])

            # 1n: union(poly1) vs poly2
            iou_1n, ol1_1n, ol2_1n = calc_metrics(g1_union, g2)
            poly1.loc[poly1['poly1_idx'].isin(p1_set), ["iou_1n", "ol_pl1_1n", "ol_pl2_1n"]] = [iou_1n, ol1_1n, ol2_1n]
            poly2.loc[poly2['poly2_idx'] == p2_set[0], ["iou_1n", "ol_pl1_1n", "ol_pl2_1n"]] = [iou_1n, ol1_1n, ol2_1n]

            # nn: each poly1 vs poly2
            for p1 in p1_set:
                g1 = poly1.loc[poly1['poly1_idx'] == p1, "geometry"].values[0]
                iou, ol1, ol2 = calc_metrics(g1, g2)
                poly1.loc[poly1['poly1_idx'] == p1, ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]
            poly2.loc[poly2['poly2_idx'] == p2_set[0], ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]

        elif rel == "N:N":
            g1_union = unary_union(poly1.loc[poly1['poly1_idx'].isin(p1_set), "geometry"])
            g2_union = unary_union(poly2.loc[poly2['poly2_idx'].isin(p2_set), "geometry"])

            # 11: union vs union
            iou_11, ol1_11, ol2_11 = calc_metrics(g1_union, g2_union)
            poly1.loc[poly1['poly1_idx'].isin(p1_set), ["iou_11", "ol_pl1_11", "ol_pl2_11"]] = [iou_11, ol1_11, ol2_11]
            poly2.loc[poly2['poly2_idx'].isin(p2_set), ["iou_11", "ol_pl1_11", "ol_pl2_11"]] = [iou_11, ol1_11, ol2_11]

            # 1n: union(poly1) vs each poly2
            for p2 in p2_set:
                g2 = poly2.loc[poly2['poly2_idx'] == p2, "geometry"].values[0]
                iou, ol1, ol2 = calc_metrics(g1_union, g2)
                poly2.loc[poly2['poly2_idx'] == p2, ["iou_1n", "ol_pl1_1n", "ol_pl2_1n"]] = [iou, ol1, ol2]
            poly1.loc[poly1['poly1_idx'].isin(p1_set), ["iou_1n", "ol_pl1_1n", "ol_pl2_1n"]] = [iou, ol1, ol2]

            # n1: each poly1 vs union(poly2)
            for p1 in p1_set:
                g1 = poly1.loc[poly1['poly1_idx'] == p1, "geometry"].values[0]
                iou, ol1, ol2 = calc_metrics(g1, g2_union)
                poly1.loc[poly1['poly1_idx'] == p1, ["iou_n1", "ol_pl1_n1", "ol_pl2_n1"]] = [iou, ol1, ol2]
            poly2.loc[poly2['poly2_idx'].isin(p2_set), ["iou_n1", "ol_pl1_n1", "ol_pl2_n1"]] = [iou, ol1, ol2]

        elif rel == "1:1":
            p1 = p1_set[0]
            p2 = p2_set[0]
            g1 = poly1.loc[poly1['poly1_idx'] == p1, "geometry"].values[0]
            g2 = poly2.loc[poly2['poly2_idx'] == p2, "geometry"].values[0]
            iou, ol1, ol2 = calc_metrics(g1, g2)
            poly1.loc[poly1['poly1_idx'] == p1, ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]
            poly2.loc[poly2['poly2_idx'] == p2, ["iou_nn", "ol_pl1_nn", "ol_pl2_nn"]] = [iou, ol1, ol2]

    return poly1, poly2


def compute_component_iou(components_dict, poly1, poly2) -> pd.DataFrame:
    records = []

    for comp_idx, comp in components_dict.items():
        poly1_idxs = comp["poly1_set"]
        poly2_idxs = comp["poly2_set"]

        if not poly1_idxs or not poly2_idxs:
            iou = 0.0
        else:
            union_poly1 = unary_union(poly1[poly1["poly1_idx"].isin(poly1_idxs)]["geometry"])
            union_poly2 = unary_union(poly2[poly2["poly2_idx"].isin(poly2_idxs)]["geometry"])

            intersection = union_poly1.intersection(union_poly2).area
            union = union_poly1.union(union_poly2).area
            iou = intersection / union if union > 0 else 0.0

        records.append({"comp_idx": comp_idx, "iou": iou})

    component_df = pd.DataFrame(records)
    return component_df


def matching_pipeline(poly1, poly2):
    # 매칭 전처리
    poly1, poly2, joined = indexing(poly1, poly2)

    # 그래프 생성
    graph = build_graph(joined)
    graph = add_energy_to_links(poly1, poly2, graph)
    component_dict, graph, cut_link, summary = split_graph_by_energy(poly1, poly2, graph, 0.05)

    # metrics 계산
    poly1, poly2 = mark_cut_links(poly1, poly2, cut_link)
    poly1, poly2 = attach_metrics_from_components(component_dict, poly1, poly2)
    component = compute_component_iou(component_dict, poly1, poly2)
    return component, poly1, poly2
