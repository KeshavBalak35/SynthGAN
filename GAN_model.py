import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

try:
    dataset = pd.read_csv(url)
    print(f"Dataset loaded successfully. Shape: {dataset.shape}")
except pd.errors.EmptyDataError:
    print("The file is empty")
    exit()
except pd.errors.ParserError:
    print("Error parsing the CSV file")
    exit()
except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit()

# Separate features and target
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

# Define the generator
def make_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(100,), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dense(32, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dense(8, activation='tanh')  # 8 features in the dataset
    ])
    return model

# Define the discriminator
def make_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(8,)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1)
    ])
    return model

# Instantiate the models
generator = make_generator()
discriminator = make_discriminator()

# Define loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Training step
@tf.function
def train_step(real_samples):
    noise = tf.random.normal([32, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_samples = generator(noise, training=True)

        real_output = discriminator(real_samples, training=True)
        fake_output = discriminator(generated_samples, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Lists to store loss values
gen_losses = []
disc_losses = []

# Function to plot losses
def plot_losses():
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.title('Generator and Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_plot_epoch_{len(gen_losses)}.png')
    plt.close()

# Training loop
def train(dataset, epochs, start_epoch=0):
    for epoch in range(start_epoch, epochs):
        epoch_gen_loss = []
        epoch_disc_loss = []
        for real_samples, _ in dataset:
            gen_loss, disc_loss = train_step(real_samples)
            epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss)

        avg_gen_loss = tf.reduce_mean(epoch_gen_loss)
        avg_disc_loss = tf.reduce_mean(epoch_disc_loss)
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
            checkpoint.save(file_prefix=checkpoint_prefix)

        if (epoch + 1) % 100 == 0:
            plot_losses()

# Function to continue training
def continue_training(dataset, additional_epochs):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Restored from checkpoint: {latest_checkpoint}")
        start_epoch = int(latest_checkpoint.split('-')[-1])
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0

    train(dataset, start_epoch + additional_epochs, start_epoch)

# Generate synthetic samples
def generate_samples(num_samples=100):
    noise = tf.random.normal([num_samples, 100])
    generated_samples = generator(noise, training=False)
    return scaler.inverse_transform(generated_samples)

# Calculate FID score
def calculate_fid(real_samples, generated_samples):
    mu1, sigma1 = real_samples.mean(axis=0), np.cov(real_samples, rowvar=False)
    mu2, sigma2 = generated_samples.mean(axis=0), np.cov(generated_samples, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)

    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Evaluate the quality of generated samples
def evaluate_samples(real_samples, generated_samples):
    real_mean = np.mean(real_samples, axis=0)
    generated_mean = np.mean(generated_samples, axis=0)

    real_std = np.std(real_samples, axis=0)
    generated_std = np.std(generated_samples, axis=0)

    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    # Create a DataFrame for easy comparison and file output
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'Real Mean': real_mean,
        'Gen Mean': generated_mean,
        'Real Std': real_std,
        'Gen Std': generated_std
    })

    # Print to console
    print("\nComparison of Real vs Generated Samples:")
    print(comparison_df.to_string(index=False))

    # Save to CSV file
    comparison_df.to_csv('feature_comparison.csv', index=False)
    print("\nFeature comparison saved to 'feature_comparison.csv'")

    fid_score = calculate_fid(real_samples, generated_samples)
    print(f"\nFrechet Inception Distance (FID) Score: {fid_score:.4f}")

    # Create plots
    create_comparison_plots(real_samples, generated_samples, feature_names)

def create_comparison_plots(real_samples, generated_samples, feature_names):
    # Histogram comparison
    fig, axes = plt.subplots(4, 2, figsize=(20, 30))
    fig.suptitle('Histograms: Real vs Generated Data', fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        sns.histplot(real_samples[:, i], kde=True, color='blue', alpha=0.5, label='Real', ax=ax)
        sns.histplot(generated_samples[:, i], kde=True, color='red', alpha=0.5, label='Generated', ax=ax)
        ax.set_title(feature_names[i])
        ax.legend()

    plt.tight_layout()
    plt.savefig('histogram_comparison.png')
    plt.close()

    # Box plot comparison
    fig, ax = plt.subplots(figsize=(15, 10))
    data_to_plot = [real_samples[:, i] for i in range(8)] + [generated_samples[:, i] for i in range(8)]
    ax.boxplot(data_to_plot, labels=feature_names + feature_names)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title('Box Plot: Real vs Generated Data')
    plt.tight_layout()
    plt.savefig('boxplot_comparison.png')
    plt.close()

    # Correlation heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(np.corrcoef(real_samples.T), annot=True, cmap='coolwarm', ax=ax1)
    ax1.set_title('Correlation Heatmap: Real Data')
    ax1.set_xticklabels(feature_names, rotation=45)
    ax1.set_yticklabels(feature_names, rotation=0)

    sns.heatmap(np.corrcoef(generated_samples.T), annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title('Correlation Heatmap: Generated Data')
    ax2.set_xticklabels(feature_names, rotation=45)
    ax2.set_yticklabels(feature_names, rotation=0)

    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

# Train the model (initial training or continue training)
train(train_dataset, epochs=5000)  # For initial training
continue_training(train_dataset, additional_epochs=2000)  # To continue training

# Generate and display samples
synthetic_samples = generate_samples(5)
print("Synthetic Samples:")
print(synthetic_samples)

# Evaluate the generated samples
real_samples = scaler.inverse_transform(X_test)
generated_samples = generate_samples(len(X_test))
evaluate_samples(real_samples, generated_samples)

# Plot final loss curves
plt.figure(figsize=(10, 5))
plt.plot(gen_losses, label='Generator Loss')
plt.plot(disc_losses, label='Discriminator Loss')
plt.title('Generator and Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('final_loss_plot.png')
plt.close()

# Save loss values to CSV
loss_df = pd.DataFrame({
    'Epoch': range(1, len(gen_losses) + 1),
    'Generator Loss': gen_losses,
    'Discriminator Loss': disc_losses
})
loss_df.to_csv('loss_values.csv', index=False)
print("Loss values saved to 'loss_values.csv'")
