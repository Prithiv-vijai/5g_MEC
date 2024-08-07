import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.models import Sequential, Model

# Load the existing dataset
data = pd.read_csv('../data/preprocessed_dataset.csv')

# Separate the Application_Type and other features
application_types = data['Application_Type']
features = data.drop(columns=['Application_Type'])

# Preprocess the features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Define the GAN
def build_generator(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(256)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    output_layer = Dense(input_dim, activation='tanh')(x)
    return Model(input_layer, output_layer)

def build_discriminator(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(1024)(input_layer)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(512)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(256)(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, output_layer)

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile the GAN
input_dim = features_scaled.shape[1]
discriminator = build_discriminator(input_dim)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

generator = build_generator(input_dim)
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Training the GAN
def train_gan(gan, generator, discriminator, epochs, batch_size, data_scaled):
    for epoch in range(epochs):
        # Train the discriminator
        noise = np.random.normal(0, 1, (batch_size, data_scaled.shape[1]))
        generated_data = generator.predict(noise)
        
        # Generate random indices using NumPy
        indices = np.random.randint(0, data_scaled.shape[0], batch_size)
        real_data = data_scaled[indices]
        
        # Convert data to float32
        real_data = real_data.astype(np.float32)
        generated_data = generated_data.astype(np.float32)
        
        combined_data = np.concatenate([real_data, generated_data], axis=0)
        
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * np.random.uniform(size=labels.shape)
        
        d_loss = discriminator.train_on_batch(combined_data, labels)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, data_scaled.shape[1]))
        misleading_labels = np.ones((batch_size, 1))
        
        g_loss = gan.train_on_batch(noise, misleading_labels)
        
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}')

# Generate synthetic data for each application type
num_synthetic_rows = 15600
num_application_types = len(application_types.unique())
rows_per_type = num_synthetic_rows // num_application_types

synthetic_data = []

for app_type in application_types.unique():
    app_type_data = data[data['Application_Type'] == app_type]
    app_type_features = app_type_data.drop(columns=['Application_Type'])
    app_type_features_scaled = scaler.fit_transform(app_type_features)
    
    # Train the GAN for this application type
    train_gan(gan, generator, discriminator, epochs=200, batch_size=64, data_scaled=app_type_features_scaled)
    
    # Generate synthetic data for this application type
    noise = np.random.normal(0, 1, (rows_per_type, app_type_features_scaled.shape[1]))
    generated_data = generator.predict(noise)
    
    # Inverse transform the data to original scale
    generated_data = scaler.inverse_transform(generated_data)
    
    # Add the application type column
    generated_data_df = pd.DataFrame(generated_data, columns=app_type_features.columns)
    generated_data_df['Application_Type'] = app_type
    
    synthetic_data.append(generated_data_df)

# Combine synthetic data for all application types
synthetic_data = pd.concat(synthetic_data)

# Combine original and synthetic data
combined_data = pd.concat([data, synthetic_data])

# Assign new User_IDs
combined_data['User_ID'] = range(1, len(combined_data) + 1)

# Save the combined dataset
combined_data.to_csv('../data/combined_dataset.csv', index=False)

# Print correlation difference
original_correlation = data.corr()
new_correlation = combined_data.corr()
correlation_diff = new_correlation - original_correlation

print("Correlation difference between original and combined datasets:")
print(correlation_diff)
