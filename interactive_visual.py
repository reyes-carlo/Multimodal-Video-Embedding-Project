import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
df = pd.read_csv('embeddings_data.csv')  # Replace with CSV file path 

# Parse text embeddings
df['text_embedding'] = df['text_embedding'].apply(eval)  # Convert stringified lists to actual lists
embeddings = np.array(df['text_embedding'].tolist())  # Convert to NumPy array

# Apply t-SNE to reduce dimensions
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Add t-SNE results to the DataFrame
df['x'] = reduced_embeddings[:, 0]
df['y'] = reduced_embeddings[:, 1]

# Initialize Dash app
app = dash.Dash(__name__)

# Create scatter plot (Change Axis names)
fig = px.scatter(df, x='x', y='y', hover_data=['text'])

# Layout
app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Div([dcc.Input(id='search-query', type='text', placeholder='Enter search query', style={'width': '70%'}),
              html.Button('Search', id='search-button'),
              ], style={'margin': '20px'}),
    html.Div(id='hover-data', style={
        'position': 'absolute',
        'top': '50px',
        'right': '50px',
        'backgroundColor': 'white',
        'padding': '10px',
        'border': '1px solid black',
        'display': 'none'  # Initially hidden
    }),
    html.Video(id='video-player', controls=True, width='600', height='400', style={'display': 'none'}) 
])


# Callback to perform semantic search
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('search-button', 'n_clicks'),
    State('search-query', 'value')
)


def semantic_search(n_clicks, query):
    if n_clicks and query:
        # Embed the query using the same embedding model (mock embedding here for illustration)
        query_embedding = np.random.rand(embeddings.shape[1])  # Replace with your model's query embedding

        # Compute cosine similarity
        similarities = cosine_similarity([query_embedding], embeddings).flatten()

        # Find top-N most similar points
        df['similarity'] = similarities
        top_n = df.nlargest(10, 'similarity')  # Adjust number of results as needed

        # Highlight the top-N points in the scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            hover_data=['text', 'similarity'],
            color=df['similarity'].apply(lambda x: 'Top Match' if x in top_n['similarity'].values else 'Other')
        )
        return fig
    return px.scatter(df)  # Default plot if no search

# Callback to display hover data in fixed position
@app.callback(
    Output('hover-data', 'children'),
    Output('hover-data', 'style'),
    Input('scatter-plot', 'hoverData')
)
def update_hover_box(hover_data):
    if hover_data:
        point_data = hover_data['points'][0]
        text = point_data.get('customdata', "No text")  # Use 'text' field or adjust based on your data
        return f"Hovered text: {text}", {'display': 'block', 'position': 'absolute', 'top': '525px', 'right': '50px',
                                         'width': '400px', 'height': '300px', 'backgroundColor': 'white',
                                         'padding': '10px', 'border': '1px solid black', 'overflowY': 'auto'} #Change position
    return "", {'display': 'none'}

# Callback to display video when a point is clicked
@app.callback(
    Output('video-player', 'src'),
    Output('video-player', 'style'),
    Input('scatter-plot', 'clickData')
)

def display_video(click_data):
    if click_data:
        file_path = click_data['points'][0]['customdata'][0]  # Get file_path from hover data
        video_url = f"https://storage.googleapis.com/{file_path}"  # Adjust as needed for GCS
        return video_url, {'display': 'block'}
    return '', {'display': 'none'}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
