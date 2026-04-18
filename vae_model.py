import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CategoricalVAE(nn.Module):
    """
    Variational Autoencoder untuk categorical features.
    Digunakan sebagai pengganti one-hot encoding.
    """
    def __init__(self, num_categories, embedding_dim=8, hidden_dim=16, latent_dim=4):
        """
        Args:
            num_categories: Jumlah kategori unik
            embedding_dim: Dimensi embedding output (untuk digunakan di diffusion)
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        super(CategoricalVAE, self).__init__()
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.linear1 = nn.Linear(num_categories, hidden_dim)
        self.lin_bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_bn2 = nn.BatchNorm1d(num_features=hidden_dim)
        
        # Latent vectors mu and sigma
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder to embedding space
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc_bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embedding_dim)
        self.fc_bn4 = nn.BatchNorm1d(embedding_dim)
        
        # Decoder to category space (untuk evaluation)
        self.fc_decode = nn.Linear(embedding_dim, num_categories)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        """
        Encode one-hot categorical ke latent space
        Args:
            x: one-hot encoded tensor [batch, num_categories]
        Returns:
            mu, logvar
        """
        h1 = self.relu(self.lin_bn1(self.linear1(x)))
        h2 = self.relu(self.lin_bn2(self.linear2(h1)))
        
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode_to_embedding(self, z):
        """
        Decode latent vector ke embedding space (untuk diffusion)
        Args:
            z: latent vector [batch, latent_dim]
        Returns:
            embedding [batch, embedding_dim]
        """
        h3 = self.relu(self.fc_bn3(self.fc3(z)))
        embedding = self.fc_bn4(self.fc4(h3))
        return embedding
    
    def decode_to_category(self, embedding):
        """
        Decode embedding ke category space (untuk evaluation)
        Args:
            embedding: [batch, embedding_dim]
        Returns:
            logits [batch, num_categories]
        """
        return self.fc_decode(embedding)
    
    def forward(self, x):
        """
        Forward pass: one-hot -> embedding
        Args:
            x: one-hot encoded [batch, num_categories]
        Returns:
            embedding, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        embedding = self.decode_to_embedding(z)
        return embedding, mu, logvar
    
    def get_embedding(self, x):
        """
        Get embedding without training mode
        Args:
            x: one-hot encoded [batch, num_categories]
        Returns:
            embedding [batch, embedding_dim]
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
            embedding = self.decode_to_embedding(mu)
        return embedding
    
    def embedding_to_category_idx(self, embedding):
        """
        Convert embedding back to category index (untuk evaluation)
        Args:
            embedding: [batch, embedding_dim]
        Returns:
            category_idx [batch]
        """
        logits = self.decode_to_category(embedding)
        return torch.argmax(logits, dim=1)


class VAELoss(nn.Module):
    """
    Custom loss untuk VAE: Reconstruction + KL Divergence
    """
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        
    def forward(self, recon_embedding, original_onehot, mu, logvar, decoder):
        """
        Args:
            recon_embedding: reconstructed embedding
            original_onehot: original one-hot encoding
            mu, logvar: latent parameters
            decoder: decoder function to convert embedding to category logits
        """
        # Reconstruction loss: decode embedding ke category space
        recon_logits = decoder(recon_embedding)
        loss_recon = F.cross_entropy(recon_logits, torch.argmax(original_onehot, dim=1), reduction='sum')
        
        # KL divergence
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return loss_recon + loss_KLD


def train_vae_for_column(cat_data_onehot, num_categories, embedding_dim=8, 
                         hidden_dim=16, latent_dim=4, epochs=500, 
                         batch_size=256, lr=1e-3, device='cpu', verbose=True):
    """
    Train VAE untuk satu categorical column
    
    Args:
        cat_data_onehot: one-hot encoded data [N, num_categories]
        num_categories: jumlah kategori
        embedding_dim: dimensi output embedding
        hidden_dim: hidden layer dimension
        latent_dim: latent dimension
        epochs: training epochs
        batch_size: batch size
        lr: learning rate
        device: torch device
        verbose: print training progress
        
    Returns:
        trained_vae_model
    """
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    
    # Convert to tensor
    if isinstance(cat_data_onehot, np.ndarray):
        cat_data_onehot = torch.from_numpy(cat_data_onehot).float()
    
    # Create dataloader
    dataset = TensorDataset(cat_data_onehot)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    vae = CategoricalVAE(num_categories, embedding_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    criterion = VAELoss()
    
    # Training loop
    vae.train()
    best_loss = float('inf')
    patience = 0
    max_patience = 50
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            
            # Forward pass
            embedding, mu, logvar = vae(x)
            
            # Compute loss
            loss = criterion(embedding, x, mu, logvar, vae.decode_to_category)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(cat_data_onehot)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}')
                break
        
        if verbose and epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    vae.eval()
    return vae
