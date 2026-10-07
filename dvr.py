import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

INFINITY = float('inf')

class DistanceVectorEngine:
    def __init__(self, nodes, edges):
        self.nodes = sorted(nodes)
        self.initial_edges = list(edges)
        self.graph = nx.Graph()
        self.neighbors = {node: {} for node in self.nodes}
        self.tables = {}
        self.reset_network()

    def reset_network(self):
        self.graph.clear()
        self.neighbors = {node: {} for node in self.nodes}
        for u, v, w in self.initial_edges:
            self.neighbors[u][v] = w
            self.neighbors[v][u] = w
            self.graph.add_edge(u, v, weight=w)

        self.tables = {}
        for node in self.nodes:
            self.tables[node] = {}
            for dest in self.nodes:
                if dest == node:
                    self.tables[node][dest] = (0, node)
                elif dest in self.neighbors[node]:
                    self.tables[node][dest] = (self.neighbors[node][dest], dest)
                else:
                    self.tables[node][dest] = (INFINITY, None)
        self.run_convergence()

    def run_convergence(self, max_rounds=20):
        rounds = 0
        while rounds < max_rounds:
            updated = False
            current_vectors = {
                node: {dest: self.tables[node][dest][0] for dest in self.nodes}
                for node in self.nodes
            }
            for x in self.nodes:
                for y in self.nodes:
                    if x == y:
                        continue
                    min_cost = self.tables[x][y][0]
                    best_hop = self.tables[x][y][1]
                    for v, cost_x_v in self.neighbors[x].items():
                        total_cost = cost_x_v + current_vectors[v][y]
                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_hop = v
                            updated = True
                    self.tables[x][y] = (min_cost, best_hop)
            rounds += 1
            if not updated:
                break
        return rounds

    def toggle_link(self, u, v):
        if self.graph.has_edge(u, v):
            # Cut link
            del self.neighbors[u][v]
            del self.neighbors[v][u]
            self.graph.remove_edge(u, v)
            if self.tables[u][v][1] == v:
                self.tables[u][v] = (INFINITY, None)
            if self.tables[v][u][1] == u:
                self.tables[v][u] = (INFINITY, None)
            is_active = False
        else:
            # Restore link from initial config
            weight = next(w for src, dst, w in self.initial_edges if (src == u and dst == v) or (src == v and dst == u))
            self.neighbors[u][v] = weight
            self.neighbors[v][u] = weight
            self.graph.add_edge(u, v, weight=weight)
            self.tables[u][v] = (weight, v)
            self.tables[v][u] = (weight, u)
            is_active = True

        self.run_convergence()
        return is_active

    def get_path(self, src, dest):
        if src == dest:
            return [src], 0
        curr = src
        path = [curr]
        total_cost = 0
        visited = set([curr])
        while curr != dest:
            cost, next_hop = self.tables[curr][dest]
            if next_hop is None or cost == INFINITY or next_hop in visited:
                return None, INFINITY
            total_cost += self.neighbors[curr][next_hop]
            curr = next_hop
            path.append(curr)
            visited.add(curr)
        return path, total_cost


class DVRDashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Distance Vector Routing (DVR) Protocol Simulator")
        self.root.geometry("1100x680")
        self.root.configure(bg="#1E1E2E")

        # Network Model
        routers = ['A', 'B', 'C', 'D', 'E']
        links = [
            ('A', 'B', 1),
            ('A', 'C', 5),
            ('B', 'C', 2),
            ('B', 'D', 4),
            ('C', 'E', 3),
            ('D', 'E', 2)
        ]
        self.engine = DistanceVectorEngine(routers, links)
        self.active_path_edges = []
        self.pos = {
            'A': (-1.0, 0.0),
            'B': (-0.3, 0.8),
            'C': (-0.3, -0.8),
            'D': (0.8, 0.6),
            'E': (0.8, -0.6)
        }

        self._build_gui()
        self.update_ui()

    def _build_gui(self):
        # Header
        header = tk.Frame(self.root, bg="#181825", height=50)
        header.pack(fill=tk.X)
        title_label = tk.Label(header, text="🌐 Dynamic Distance Vector Routing Simulator",
                               font=("Segoe UI", 16, "bold"), fg="#CDD6F4", bg="#181825", pady=10)
        title_label.pack()

        # Main Body Splitter
        main_frame = tk.Frame(self.root, bg="#1E1E2E")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left Column: Graph Visualization
        left_frame = tk.Frame(main_frame, bg="#181825", bd=2, relief=tk.RIDGE)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.fig, self.ax = plt.subplots(figsize=(6, 5), facecolor="#181825")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right Column: Controls & Routing Tables
        right_frame = tk.Frame(main_frame, bg="#1E1E2E", width=420)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))

        # 1. Action Controls Card
        ctrl_card = tk.LabelFrame(right_frame, text=" Simulation Controls ", font=("Segoe UI", 11, "bold"),
                                  fg="#89B4FA", bg="#181825", bd=1, padx=10, pady=10)
        ctrl_card.pack(fill=tk.X, pady=(0, 10))

        self.fail_btn = tk.Button(ctrl_card, text="💥 Sever Link (B <-> C)", font=("Segoe UI", 10, "bold"),
                                  bg="#F38BA8", fg="#11111B", activebackground="#EBA0AC",
                                  command=self.toggle_bc_link, relief=tk.FLAT, padx=10, pady=5)
        self.fail_btn.pack(fill=tk.X, pady=3)

        reset_btn = tk.Button(ctrl_card, text="🔄 Reset All Links", font=("Segoe UI", 10),
                              bg="#A6ADC8", fg="#11111B", command=self.reset_all, relief=tk.FLAT, pady=4)
        reset_btn.pack(fill=tk.X, pady=3)

        # 2. Packet Tracer Card
        trace_card = tk.LabelFrame(right_frame, text=" Live Packet Transmission Tracer ", font=("Segoe UI", 11, "bold"),
                                   fg="#A6E3A1", bg="#181825", bd=1, padx=10, pady=8)
        trace_card.pack(fill=tk.X, pady=(0, 10))

        trace_inputs = tk.Frame(trace_card, bg="#181825")
        trace_inputs.pack(fill=tk.X, pady=2)

        tk.Label(trace_inputs, text="Source:", fg="#CDD6F4", bg="#181825", font=("Segoe UI", 10)).grid(row=0, column=0, padx=4)
        self.src_combo = ttk.Combobox(trace_inputs, values=self.engine.nodes, width=4, state="readonly")
        self.src_combo.set('A')
        self.src_combo.grid(row=0, column=1, padx=4)

        tk.Label(trace_inputs, text="Destination:", fg="#CDD6F4", bg="#181825", font=("Segoe UI", 10)).grid(row=0, column=2, padx=4)
        self.dest_combo = ttk.Combobox(trace_inputs, values=self.engine.nodes, width=4, state="readonly")
        self.dest_combo.set('E')
        self.dest_combo.grid(row=0, column=3, padx=4)

        send_btn = tk.Button(trace_inputs, text="🚀 Send Packet", font=("Segoe UI", 9, "bold"),
                             bg="#A6E3A1", fg="#11111B", command=self.trace_packet, relief=tk.FLAT, padx=6)
        send_btn.grid(row=0, column=4, padx=8)

        self.trace_status = tk.Label(trace_card, text="Status: Ready", font=("Segoe UI", 9, "italic"),
                                     fg="#BAC2DE", bg="#181825", wraplength=380, justify=tk.LEFT)
        self.trace_status.pack(fill=tk.X, pady=(6, 2))

        # 3. Routing Table Viewer Card
        table_card = tk.LabelFrame(right_frame, text=" Live Routing Table Inspector ", font=("Segoe UI", 11, "bold"),
                                   fg="#F9E2AF", bg="#181825", bd=1, padx=10, pady=8)
        table_card.pack(fill=tk.BOTH, expand=True)

        sel_frame = tk.Frame(table_card, bg="#181825")
        sel_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(sel_frame, text="View Table for Router:", fg="#CDD6F4", bg="#181825", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=4)
        self.node_sel_combo = ttk.Combobox(sel_frame, values=self.engine.nodes, width=5, state="readonly")
        self.node_sel_combo.set('A')
        self.node_sel_combo.pack(side=tk.LEFT, padx=4)
        self.node_sel_combo.bind("<<ComboboxSelected>>", lambda e: self.render_table())

        # Treeview Table
        cols = ("Destination", "Cost", "Next Hop")
        self.tree = ttk.Treeview(table_card, columns=cols, show='headings', height=6)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def toggle_bc_link(self):
        is_active = self.engine.toggle_link('B', 'C')
        if not is_active:
            self.fail_btn.config(text="🩹 Restore Link (B <-> C)", bg="#F9E2AF")
        else:
            self.fail_btn.config(text="💥 Sever Link (B <-> C)", bg="#F38BA8")
        self.active_path_edges = []
        self.trace_status.config(text="Link state toggled. Routes recomputed.")
        self.update_ui()

    def reset_all(self):
        self.engine.reset_network()
        self.active_path_edges = []
        self.fail_btn.config(text="💥 Sever Link (B <-> C)", bg="#F38BA8")
        self.trace_status.config(text="Network reset to initial state.")
        self.update_ui()

    def trace_packet(self):
        src = self.src_combo.get()
        dest = self.dest_combo.get()
        path, cost = self.engine.get_path(src, dest)
        if path is None:
            self.active_path_edges = []
            self.trace_status.config(text=f"❌ No route exists from [{src}] to [{dest}]!", fg="#F38BA8")
        else:
            self.active_path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            path_str = " -> ".join(path)
            self.trace_status.config(text=f"✅ Packet Routed: {path_str} (Cost: {cost})", fg="#A6E3A1")
        self.draw_graph()

    def render_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        node = self.node_sel_combo.get()
        for dest in self.engine.nodes:
            cost, hop = self.engine.tables[node][dest]
            cost_str = str(cost) if cost != INFINITY else "INF"
            hop_str = str(hop) if hop is not None else "-"
            self.tree.insert("", tk.END, values=(dest, cost_str, hop_str))

    def draw_graph(self):
        self.ax.clear()
        self.ax.set_facecolor("#181825")

        # Edges
        all_edges = list(self.engine.graph.edges())
        path_set = set(tuple(sorted(e)) for e in self.active_path_edges)

        normal_edges = [e for e in all_edges if tuple(sorted(e)) not in path_set]
        active_edges = [e for e in all_edges if tuple(sorted(e)) in path_set]

        # Draw base edges
        nx.draw_networkx_edges(self.engine.graph, self.pos, edgelist=normal_edges,
                               width=2.5, edge_color='#585B70', ax=self.ax)
        # Highlight packet path in Neon Green
        if active_edges:
            nx.draw_networkx_edges(self.engine.graph, self.pos, edgelist=active_edges,
                                   width=5.0, edge_color='#A6E3A1', ax=self.ax)

        # Draw Nodes
        nx.draw_networkx_nodes(self.engine.graph, self.pos, node_color='#89B4FA',
                               node_size=1100, ax=self.ax)
        nx.draw_networkx_labels(self.engine.graph, self.pos, font_color='#11111B',
                                font_weight='bold', font_size=11, ax=self.ax)

        # Draw Weights
        edge_labels = nx.get_edge_attributes(self.engine.graph, 'weight')
        nx.draw_networkx_edge_labels(self.engine.graph, self.pos, edge_labels=edge_labels,
                                     font_size=10, font_color='#F9E2AF', font_weight='bold',
                                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#181825', edgecolor='none'),
                                     ax=self.ax)

        self.ax.axis('off')
        self.canvas.draw()

    def update_ui(self):
        self.draw_graph()
        self.render_table()


if __name__ == "__main__":
    root = tk.Tk()
    app = DVRDashboardApp(root)
    root.mainloop()