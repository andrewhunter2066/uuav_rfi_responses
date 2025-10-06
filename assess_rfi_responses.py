import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model

# Sample responses (simulating responses from an RFI survey)
df = pd.read_csv('./testing/responses.csv')
texts = df['response']

# Convert responses into a document-term matrix
vectorizer = CountVectorizer(stop_words='english', max_features=50)
dtm = vectorizer.fit_transform(texts)

# Fit the LDA model
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(dtm)

# Extract and print top terms for each topic
num_top_terms = 5
terms = vectorizer.get_feature_names_out()

print("\nMost significant terms per topic:")
for idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-num_top_terms:][::-1]
    top_terms = [terms[i] for i in top_indices]
    print(f"\nTopic {idx + 1}: {', '.join(top_terms)}")

# Generate and save visualisations
vis = pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)
pyLDAvis.save_html(vis, 'lda_visualisation.html')
print("\nInteractive topic visualization saved to 'lda_visualization.html'. Open this file in a browser to explore the topics.")