import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import os
import csv

# Initialize Dash app
app = dash.Dash(__name__)

# Define custom CSS styling
app.clientside_callback(
    """
    function(n_clicks) {
        return {'display': 'block', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f4f4f9', 'borderRadius': '10px'};
    }
    """,
    Output('feedback', 'style'),
    [Input('add-user-button', 'n_clicks')]
)

class MECEnvironment:
    def __init__(self, num_nodes, bandwidth):
        self.num_nodes = num_nodes
        self.bandwidth = bandwidth
        self.nodes = list(range(num_nodes))
        self.graph = nx.Graph()
        self.user_file = 'users.csv'
        self._initialize_nodes()

    def _initialize_nodes(self):
        # Fixed positions for edge nodes
        positions = [
            (10, 10),
            (90, 10),
            (10, 90),
            (90, 90)
        ]
        for node, pos in zip(self.nodes, positions):
            self.graph.add_node(node, pos=pos, bandwidth=self.bandwidth, users=0)

    def add_user(self, user_id, x, y, required_bandwidth):
        if user_id in self.graph.nodes:
            raise ValueError(f"User ID {user_id} already exists.")
        nearest_node = self._find_nearest_node(x, y)
        if nearest_node is None:
            raise ValueError("No nearest node found.")
        pos = self.graph.nodes[nearest_node]['pos']
        distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        if distance > 50:
            raise ValueError("User is too far from any edge node.")
        self.graph.add_node(user_id, pos=(x, y), required_bandwidth=required_bandwidth)
        self.graph.nodes[nearest_node]['bandwidth'] -= required_bandwidth
        self.graph.nodes[nearest_node]['users'] += 1
        self.save_users()

    def _find_nearest_node(self, x, y):
        pos = nx.get_node_attributes(self.graph, 'pos')
        distances = {node: np.sqrt((x - pos[node][0])**2 + (y - pos[node][1])**2) for node in self.nodes}
        return min(distances, key=distances.get)

    def save_users(self):
        with open(self.user_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'pos_x', 'pos_y', 'required_bandwidth'])
            for node, data in self.graph.nodes(data=True):
                if isinstance(node, str):  # User nodes
                    writer.writerow([node, data['pos'][0], data['pos'][1], data['required_bandwidth']])

    def load_users(self):
        if os.path.exists(self.user_file):
            with open(self.user_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_id = row['user_id']
                    pos_x = int(row['pos_x'])
                    pos_y = int(row['pos_y'])
                    required_bandwidth = int(row['required_bandwidth'])
                    self.graph.add_node(user_id, pos=(pos_x, pos_y), required_bandwidth=required_bandwidth)
                    nearest_node = self._find_nearest_node(pos_x, pos_y)
                    self.graph.nodes[nearest_node]['bandwidth'] -= required_bandwidth
                    self.graph.nodes[nearest_node]['users'] += 1

    def plot_graph(self):
        pos = nx.get_node_attributes(self.graph, 'pos')
        node_colors = ['#1f77b4' if isinstance(node, int) else '#ff7f0e' for node in self.graph.nodes()]
        labels = {node: f"Node {node}\nBandwidth: {self.graph.nodes[node].get('bandwidth', 0)}\nUsers: {self.graph.nodes[node].get('users', 0)}" for node in self.graph.nodes() if isinstance(node, int)}
        user_labels = {node: f"User {node}" for node in self.graph.nodes() if not isinstance(node, int)}
        labels.update(user_labels)

        edge_x = []
        edge_y = []
        for node in self.graph.nodes:
            if not isinstance(node, int):  # User nodes
                user_pos = pos[node]
                nearest_node = self._find_nearest_node(user_pos[0], user_pos[1])
                edge_pos = pos[nearest_node]
                edge_x.extend([user_pos[0], edge_pos[0], None])
                edge_y.extend([user_pos[1], edge_pos[1], None])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='black', dash='dash'), name='User Connections'))

        node_x = [pos[node][0] for node in self.graph.nodes()]
        node_y = [pos[node][1] for node in self.graph.nodes()]
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[labels.get(node, '') for node in self.graph.nodes()],
                                 marker=dict(size=12, color=node_colors, line=dict(color='black', width=2)), name='Nodes'))

        fig.update_layout(title='MEC Environment', showlegend=False, xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False),
                          plot_bgcolor='#eaeaea', paper_bgcolor='#ffffff')

        return fig

    def get_node_details(self):
        details = []
        for node in self.graph.nodes:
            if isinstance(node, int):  # Edge nodes
                node_data = self.graph.nodes[node]
                details.append({
                    'Node': node,
                    'Available Bandwidth': node_data.get('bandwidth', 0),
                    'Connected Users': node_data.get('users', 0)
                })
        return details

# Initialize MEC Environment
num_nodes = 4
bandwidth = 1000
mec_env = MECEnvironment(num_nodes, bandwidth)

# Dash Layout
app.layout = html.Div([
    html.Div([
        html.H1('MEC Environment Simulation', style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#333', 'font-family': 'Arial, sans-serif'}),
        
        html.Div([
            dcc.Graph(id='mec-graph', figure=mec_env.plot_graph(), style={'height': '60vh', 'width': '100%'})
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.H3('Add New User', style={'textAlign': 'center', 'marginBottom': '10px', 'color': '#333', 'font-family': 'Arial, sans-serif'}),
            dcc.Input(id='user-id', type='text', placeholder='User ID', style={'margin': '5px', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            dcc.Input(id='x-coord', type='number', placeholder='X Coordinate', min=0, max=100, style={'margin': '5px', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            dcc.Input(id='y-coord', type='number', placeholder='Y Coordinate', min=0, max=100, style={'margin': '5px', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            dcc.Input(id='bandwidth', type='number', placeholder='Required Bandwidth', min=0, style={'margin': '5px', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
            html.Button('Add User', id='add-user-button', n_clicks=0, style={'margin': '5px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'textAlign': 'center', 'textDecoration': 'none', 'display': 'inline-block', 'fontSize': '16px', 'borderRadius': '5px', 'cursor': 'pointer'}),
            html.Button('Load Existing Users', id='load-users-button', n_clicks=0, style={'margin': '5px', 'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'textAlign': 'center', 'textDecoration': 'none', 'display': 'inline-block', 'fontSize': '16px', 'borderRadius': '5px', 'cursor': 'pointer'}),
            html.Div(id='feedback', style={'textAlign': 'center', 'marginTop': '10px', 'color': '#e74c3c', 'font-family': 'Arial, sans-serif'})
        ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.2)', 'marginBottom': '20px'}),

        html.Div(id='node-details', style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'})
    ], style={'backgroundColor': '#f4f4f9', 'padding': '20px'})
])

@app.callback(
    [Output('mec-graph', 'figure'),
     Output('node-details', 'children')],
    [Input('add-user-button', 'n_clicks'),
     Input('load-users-button', 'n_clicks')],
    [State('user-id', 'value'),
     State('x-coord', 'value'),
     State('y-coord', 'value'),
     State('bandwidth', 'value')]
)
def update_content(add_n_clicks, load_n_clicks, user_id, x, y, bandwidth):
    if add_n_clicks > 0:
        try:
            mec_env.add_user(user_id, x, y, bandwidth)
            feedback = f"User {user_id} added successfully!"
        except ValueError as e:
            feedback = str(e)
    elif load_n_clicks > 0:
        mec_env.load_users()
        feedback = "Existing users loaded successfully!"
    
    # Generate node details
    node_details = mec_env.get_node_details()
    details_table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in node_details[0].keys()]), style={'backgroundColor': '#f2f2f2'}),
        html.Tbody([
            html.Tr([html.Td(row[col]) for col in row], style={'borderBottom': '1px solid #ddd'}) for row in node_details
        ])
    ]) if node_details else html.P("No nodes available")

    return mec_env.plot_graph(), details_table

if __name__ == '__main__':
    app.run_server(debug=False)
