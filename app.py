import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

# Connexion à la base
conn = sqlite3.connect("ventes.db")

# Chargement des données
clients = pd.read_sql_query("SELECT * FROM clients", conn)
produits = pd.read_sql_query("SELECT * FROM produits", conn)
ventes = pd.read_sql_query("""
    SELECT v.id_vente, c.nom AS client, c.region, p.nom AS produit,
           p.prix, v.quantite, v.date_vente, (v.quantite*p.prix) AS CA
    FROM ventes v
    JOIN clients c ON v.id_client = c.id_client
    JOIN produits p ON v.id_produit = p.id_produit
""", conn)

st.title("📊 Tableau de bord BI – Numilog")

# KPI globaux
st.metric("💰 Chiffre d’affaires total", f"{ventes['CA'].sum():,.0f} DZD")
st.metric("📦 Quantité totale vendue", f"{ventes['quantite'].sum()} unités")

# Graphique CA par région
fig1 = px.bar(ventes.groupby('region')['CA'].sum().reset_index(),
              x='region', y='CA', title="CA par région", color='region')
st.plotly_chart(fig1)

# Top produits
fig2 = px.pie(ventes.groupby('produit')['CA'].sum().reset_index(),
              values='CA', names='produit', title="Top produits")
st.plotly_chart(fig2)

# Évolution mensuelle du CA
ventes['date_vente'] = pd.to_datetime(ventes['date_vente'])
ventes['mois'] = ventes['date_vente'].dt.to_period('M')
fig3 = px.line(ventes.groupby('mois')['CA'].sum().reset_index(),
               x='mois', y='CA', title="Évolution mensuelle du CA")
st.plotly_chart(fig3)
