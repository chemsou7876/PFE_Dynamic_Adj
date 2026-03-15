# ============================================================
# dynamic_adjacency.py
# Layer 2 : Calcul de l'adjacence dynamique par fenêtre
# ============================================================

import numpy as np
import torch
from pathlib import Path


class DynamicAdjacency:
    """
    Calcule A_dynamic pour chaque fenêtre temporelle.
    
    Utilise 3 ingrédients :
      1. A_static (from Neo4j/Layer 1) — prior spatial GPS
      2. quality_score — fiabilité du capteur dans cette fenêtre
      3. local_corr — corrélation temporelle locale entre capteurs
    """

    def __init__(self, data_dir="./extracted_data", alpha=0.6, beta=0.4, device=None):
        """
        Args:
            data_dir: dossier contenant A_static.npy, var_mean.npy, var_std.npy
            alpha: poids du prior spatial (GPS distance)
            beta: poids du contexte temporel (corrélation locale)
            device: 'cuda:0' ou 'cpu'
        """
        data_dir = Path(data_dir)
        
        # Charger les données de Layer 1
        self.A_static = torch.tensor(np.load(data_dir / "A_static.npy")).float()
        self.var_mean = torch.tensor(np.load(data_dir / "var_mean.npy")).float()
        self.var_std = torch.tensor(np.load(data_dir / "var_std.npy")).float()

        self.alpha = alpha
        self.beta = beta
        self.device = device or torch.device('cpu')

        # Envoyer sur le bon device
        self.A_static = self.A_static.to(self.device)
        self.var_mean = self.var_mean.to(self.device)
        self.var_std = self.var_std.to(self.device)

        # Seuils d'anomalie : μ + 3σ
        self.anomaly_threshold = self.var_mean + 3 * self.var_std

        self.n_sensors = self.A_static.shape[0]

        print(f"✅ DynamicAdjacency initialisée")
        print(f"   {self.n_sensors} capteurs, α={alpha}, β={beta}")
        print(f"   A_static non-zero: {(self.A_static > 0).sum().item()}")

    def compute_quality_score(self, observed_mask):
        """
        Étape 2a : Calcul du quality score par capteur.
        
        Args:
            observed_mask: (B, K, L) — 1 si observé, 0 si manquant
            
        Returns:
            quality: (B, K) — score entre 0 et 1 par capteur
        """
        B, K, L = observed_mask.shape

        # 1. Availability = proportion de valeurs observées
        availability = observed_mask.mean(dim=-1)  # (B, K)

        # 2. Anomaly detection (on ne peut pas la calculer ici car on n'a pas
        #    les vraies valeurs pour les données manquantes, donc on utilise
        #    seulement l'availability pour l'instant)
        #    Note: en version complète, on comparerait la variance locale
        #    avec self.anomaly_threshold

        # quality_score = availability (simplifié pour le premier test)
        quality = availability  # (B, K)

        return quality

    def compute_local_correlation(self, observed_data, observed_mask):
        """
        Étape 2b : Corrélation de Pearson locale entre capteurs.
        
        Args:
            observed_data: (B, K, L) — données observées (normalisées)
            observed_mask: (B, K, L) — masque d'observation
            
        Returns:
            corr: (B, K, K) — matrice de corrélation locale
        """
        B, K, L = observed_data.shape

        # Masquer les données non observées
        masked_data = observed_data * observed_mask  # (B, K, L)

        # Nombre de co-observations par paire
        co_obs = torch.bmm(observed_mask, observed_mask.transpose(1, 2))  # (B, K, K)

        # Somme des valeurs observées
        sum_x = masked_data.sum(dim=-1, keepdim=True)  # (B, K, 1)

        # Moyenne (seulement sur les valeurs observées)
        n_obs = observed_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, K, 1)
        mean_x = sum_x / n_obs  # (B, K, 1)

        # Centrer les données
        centered = (masked_data - mean_x) * observed_mask  # (B, K, L)

        # Covariance
        cov = torch.bmm(centered, centered.transpose(1, 2))  # (B, K, K)
        cov = cov / co_obs.clamp(min=1)

        # Écarts-types
        std = torch.sqrt((centered ** 2).sum(dim=-1) / n_obs.squeeze(-1).clamp(min=1))  # (B, K)
        std = std.clamp(min=1e-8)

        # Corrélation = cov / (std_i × std_j)
        std_outer = std.unsqueeze(-1) * std.unsqueeze(-2)  # (B, K, K)
        corr = cov / std_outer.clamp(min=1e-8)

        # Clamper entre 0 et 1 (on ne veut pas de corrélations négatives)
        corr = corr.clamp(0, 1)

        # Mettre la diagonale à 0
        eye = torch.eye(K, device=self.device).unsqueeze(0)
        corr = corr * (1 - eye)

        return corr  # (B, K, K)

    def compute_dynamic_adj(self, observed_data, observed_mask):
        """
        Étape 2c : Construction de A_dynamic.
        
        La formule complète :
        A_dynamic(i,j,t) = [α × A_static(i,j) + β × local_corr(i,j,t)]
                           × quality_score(j, t)
        
        Args:
            observed_data: (B, K, L) — données observées
            observed_mask: (B, K, L) — masque d'observation
            
        Returns:
            adj_dynamic: (B, K, K) — adjacence dynamique par batch
        """
        B, K, L = observed_data.shape

        # Étape 2a : Quality score
        quality = self.compute_quality_score(observed_mask)  # (B, K)

        # Étape 2b : Corrélation locale
        local_corr = self.compute_local_correlation(observed_data, observed_mask)  # (B, K, K)

        # Étape 2c : Fusion
        # A_static est (K, K), on l'expand pour le batch
        A_static_batch = self.A_static.unsqueeze(0).expand(B, -1, -1)  # (B, K, K)

        # Combinaison : α × A_static + β × local_corr
        adj_combined = self.alpha * A_static_batch + self.beta * local_corr  # (B, K, K)

        # Quality gate : on multiplie par le quality score du capteur SOURCE (colonne j)
        quality_gate = quality.unsqueeze(1).expand(-1, K, -1)  # (B, K, K)
        # quality_gate[b, i, j] = quality[b, j]

        adj_dynamic = adj_combined * quality_gate  # (B, K, K)

        return adj_dynamic


# ============================================================
# TEST : Vérifier que tout fonctionne
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST DynamicAdjacency")
    print("=" * 50)

    # Créer l'instance
    dyn_adj = DynamicAdjacency(data_dir="./extracted_data", alpha=0.6, beta=0.4)

    # Simuler un batch de données
    B, K, L = 2, 36, 36  # 2 samples, 36 capteurs, 36 timesteps
    torch.manual_seed(42)

    # Données aléatoires normalisées
    observed_data = torch.randn(B, K, L)

    # Masque : la plupart des capteurs sont observés, sauf quelques-uns
    observed_mask = torch.ones(B, K, L)
    # Capteur 5 : 90% manquant dans le batch 0
    observed_mask[0, 5, :] = 0.
    observed_mask[0, 5, 0:3] = 1.  # seulement 3 valeurs sur 36
    # Capteur 10 : complètement manquant dans le batch 1
    observed_mask[1, 10, :] = 0.

    # Calculer A_dynamic
    A_dynamic = dyn_adj.compute_dynamic_adj(observed_data, observed_mask)

    print(f"\nA_dynamic shape: {A_dynamic.shape}")  # (2, 36, 36)

    # Vérifier les comportements attendus
    print(f"\n── Batch 0 ──")
    print(f"Quality capteur 5  : {dyn_adj.compute_quality_score(observed_mask)[0, 5]:.4f}")
    print(f"Quality capteur 0  : {dyn_adj.compute_quality_score(observed_mask)[0, 0]:.4f}")
    print(f"A_dynamic[0, 0, 5] : {A_dynamic[0, 0, 5]:.4f}  (devrait être faible)")
    print(f"A_dynamic[0, 0, 1] : {A_dynamic[0, 0, 1]:.4f}  (devrait être normal)")
    print(f"A_static[0, 5]     : {dyn_adj.A_static[0, 5]:.4f}  (référence)")
    print(f"A_static[0, 1]     : {dyn_adj.A_static[0, 1]:.4f}  (référence)")

    print(f"\n── Batch 1 ──")
    print(f"Quality capteur 10 : {dyn_adj.compute_quality_score(observed_mask)[1, 10]:.4f}")
    print(f"A_dynamic[1, 0, 10]: {A_dynamic[1, 0, 10]:.4f}  (devrait être ~0)")

    # Comparer avec A_static
    print(f"\n── Comparaison avec A_static ──")
    A_static_val = dyn_adj.A_static[0, 1].item()
    A_dynamic_val = A_dynamic[0, 0, 1].item()
    print(f"A_static[0,1]      = {A_static_val:.4f}")
    print(f"A_dynamic[0,0,1]   = {A_dynamic_val:.4f}")
    print(f"Ratio              = {A_dynamic_val / A_static_val:.4f}  (devrait être ~α×1.0 + β×corr)")

    print("\n🎉 Test terminé !")