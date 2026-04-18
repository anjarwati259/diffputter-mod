import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CategoricalVAE(nn.Module):
    """
    Improved Variational Autoencoder untuk categorical features.
    Arsitektur yang lebih dalam untuk representasi yang lebih baik.
    """
    def __init__(self, num_categories, embedding_dim=8, hidden_dim=None, latent_dim=None):
        """
        Args:
            num_categories: Jumlah kategori unik
            embedding_dim: Dimensi embedding output (untuk digunakan di diffusion)
            hidden_dim: Hidden layer dimension (auto-scale jika None)
            latent_dim: Latent space dimension (auto-scale jika None)
        """
        super(CategoricalVAE, self).__init__()
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        
        # Auto-scale hidden dimensions based on num_categories and embedding_dim
        if hidden_dim is None:
            hidden_dim = max(32, num_categories * 2, embedding_dim * 4)
        if latent_dim is None:
            latent_dim = max(4, embedding_dim // 2)
            
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder - Deeper network with residual-like connections
        self.enc1 = nn.Linear(num_categories, hidden_dim)
        self.enc_bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc_dropout1 = nn.Dropout(0.1)
        
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_bn2 = nn.BatchNorm1d(hidden_dim)
        self.enc_dropout2 = nn.Dropout(0.1)
        
        self.enc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.enc_bn3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Latent vectors mu and logvar
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder to embedding space - Mirror encoder structure
        self.dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.dec_bn1 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.dec_bn2 = nn.BatchNorm1d(hidden_dim)
        self.dec_dropout2 = nn.Dropout(0.1)
        
        self.dec3 = nn.Linear(hidden_dim, embedding_dim)
        self.dec_bn3 = nn.BatchNorm1d(embedding_dim)
        
        # Decoder to category space (untuk evaluation)
        self.fc_decode = nn.Linear(embedding_dim, num_categories)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def encode(self, x):
        """
        Encode one-hot categorical ke latent space dengan deeper network
        Args:
            x: one-hot encoded tensor [batch, num_categories]
        Returns:
            mu, logvar
        """
        # Layer 1
        h1 = self.leaky_relu(self.enc_bn1(self.enc1(x)))
        h1 = self.enc_dropout1(h1)
        
        # Layer 2 with residual-like connection
        h2 = self.leaky_relu(self.enc_bn2(self.enc2(h1)))
        h2 = self.enc_dropout2(h2)
        h2 = h2 + h1  # Residual connection
        
        # Layer 3
        h3 = self.leaky_relu(self.enc_bn3(self.enc3(h2)))
        
        # Latent parameters
        mu = self.fc_mu(h3)
        logvar = self.fc_logvar(h3)
        
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
        Decode latent vector ke embedding space dengan deeper network
        Args:
            z: latent vector [batch, latent_dim]
        Returns:
            embedding [batch, embedding_dim]
        """
        # Mirror encoder structure
        d1 = self.leaky_relu(self.dec_bn1(self.dec1(z)))
        d2 = self.leaky_relu(self.dec_bn2(self.dec2(d1)))
        d2 = self.dec_dropout2(d2)
        
        # Final embedding layer
        embedding = self.dec_bn3(self.dec3(d2))
        
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