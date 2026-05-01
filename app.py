import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import cv2
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------
# Paths & params
# --------------------
DATA_PATH = "Small-80"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
TEST_DIR  = os.path.join(DATA_PATH, "test")
METADATA_PATH = "Small-80\metadata.csv"
MODEL_PATH = "densenet201_metadata_final.h5"
IMG_SIZE = 224

# --------------------
# Load data and model
# --------------------
@st.cache_resource
def load_data_and_model():
    """Load metadata and trained model"""
    df = pd.read_csv(METADATA_PATH)
    
    if not df['image_id'].iloc[0].lower().endswith('.jpg'):
        df['image_id'] = df['image_id'].astype(str) + ".jpg"

    df['image_path'] = df.apply(
        lambda r: os.path.join(TRAIN_DIR if r['split']=='train' else TEST_DIR, r['label'], r['image_id']),
        axis=1
    )

    exists_mask = df['image_path'].apply(os.path.exists)
    df = df[exists_mask].reset_index(drop=True)

    label_encoder = LabelEncoder()
    df['label_enc'] = label_encoder.fit_transform(df['label'])
    num_classes = df['label_enc'].nunique()

    variety_encoder = LabelEncoder()
    df['variety_enc'] = variety_encoder.fit_transform(df['variety'])
    num_varieties = df['variety_enc'].nunique()

    df['age_norm'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())

    # Load the trained model
    model = load_model(MODEL_PATH)
    
    return df, label_encoder, variety_encoder, model

# --------------------
# GradCAM Implementation
# --------------------
class GradCAM:
    def __init__(self, model, layer_name="conv5_block32_concat"):
        self.model = model
        self.layer_name = layer_name
        
        # Try to find a suitable convolutional layer
        conv_layers = [layer.name for layer in model.layers 
                      if 'conv' in layer.name or 'concat' in layer.name]
        if not conv_layers:
            # If no conv layers found, use the last layer before flattening
            for layer in model.layers:
                if len(layer.output_shape) > 2:  # Has spatial dimensions
                    self.layer_name = layer.name
                    break
        
        st.info(f"Using layer for GradCAM: {self.layer_name}")
        
        self.grad_model = tf.keras.models.Model(
            inputs=[model.inputs[0], model.inputs[1], model.inputs[2]],
            outputs=[model.get_layer(self.layer_name).output, model.output]
        )
    
    def generate_heatmap(self, img_array, variety, age, class_idx=None):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model([img_array, variety, age])
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            st.warning("Gradients are None. Using uniform heatmap.")
            heatmap = np.ones(conv_outputs.shape[1:3])
            return heatmap, class_idx.numpy()
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy(), class_idx.numpy()
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4):
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * alpha + img
        return np.clip(superimposed_img, 0, 255).astype(np.uint8)

def create_gradcam_visualization(model, image, variety_enc, age_norm, label_encoder):
    """Create GradCAM visualization for a single image"""
    grad_cam = GradCAM(model)
    
    # Prepare input for model
    img_array = tf.expand_dims(image, 0).numpy().astype(np.float32) / 255.0
    variety_array = np.array([[variety_enc]], dtype=np.int32)
    age_array = np.array([[age_norm]], dtype=np.float32)
    
    # Generate heatmap
    heatmap, pred_class = grad_cam.generate_heatmap(img_array, variety_array, age_array)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("GradCAM Heatmap")
    axes[1].axis('off')
    
    # Overlay
    superimposed_img = grad_cam.overlay_heatmap(image, heatmap)
    axes[2].imshow(superimposed_img)
    axes[2].set_title(f"Overlay\nPredicted: {label_encoder.classes_[pred_class]}")
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig, pred_class

# --------------------
# Feature Importance Analysis
# --------------------
def create_feature_importance_analysis(model, image, variety_enc, age_norm, num_varieties, label_encoder, pred_class):
    """Create feature importance analysis for the prediction"""
    
    # Prepare base input
    img_array = tf.expand_dims(image, 0).numpy().astype(np.float32) / 255.0
    variety_array = np.array([[variety_enc]], dtype=np.int32)
    age_array = np.array([[age_norm]], dtype=np.float32)
    
    # Base prediction
    base_pred = model.predict([img_array, variety_array, age_array], verbose=0)
    base_confidence = base_pred[0][pred_class]
    
    # Test variety impact
    variety_impacts = []
    variety_probs = []
    for test_variety in range(num_varieties):
        test_pred = model.predict([img_array,
                                 np.array([[test_variety]], dtype=np.int32),
                                 age_array], verbose=0)
        impact = np.abs(test_pred - base_pred).mean()
        variety_impacts.append(impact)
        variety_probs.append(test_pred[0][pred_class])
    
    # Test age impact
    age_impacts = []
    age_probs = []
    test_ages = np.linspace(0, 1, 5)
    for test_age in test_ages:
        test_pred = model.predict([img_array,
                                 variety_array,
                                 np.array([[test_age]], dtype=np.float32)], verbose=0)
        impact = np.abs(test_pred - base_pred).mean()
        age_impacts.append(impact)
        age_probs.append(test_pred[0][pred_class])
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature importance bar chart
    features = ['Variety', 'Age']
    importance_means = [np.mean(variety_impacts), np.mean(age_impacts)]
    
    bars = ax1.bar(features, importance_means, color=['skyblue', 'lightcoral'], alpha=0.7)
    ax1.set_ylabel('Average Prediction Impact')
    ax1.set_title('Metadata Feature Importance')
    ax1.grid(axis='y', alpha=0.3) 
    
    for bar, value in zip(bars, importance_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Variety impact on prediction
    variety_names = [f'Var_{i}' for i in range(len(variety_probs))]
    ax2.bar(variety_names, variety_probs, color='orange', alpha=0.7)
    ax2.axhline(y=base_confidence, color='red', linestyle='--', label='Current Variety')
    ax2.set_title('Variety Impact on Prediction Confidence')
    ax2.set_ylabel('Predicted Class Probability')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Age impact on prediction
    ax3.plot(test_ages, age_probs, 'o-', color='green', linewidth=2, markersize=8)
    ax3.axhline(y=base_confidence, color='red', linestyle='--', label='Current Age')
    ax3.set_xlabel('Age (Normalized)')
    ax3.set_ylabel('Predicted Class Probability')
    ax3.set_title('Age Impact on Prediction Confidence')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Confidence contribution breakdown
    confidence_components = {
        'Image Features': base_confidence * 0.6,
        'Variety': (max(variety_probs) - min(variety_probs)) * 0.2,
        'Age': (max(age_probs) - min(age_probs)) * 0.2
    }
    
    colors = ['blue', 'orange', 'green']
    ax4.bar(confidence_components.keys(), confidence_components.values(), 
           color=colors, alpha=0.7)
    ax4.set_ylabel('Confidence Contribution')
    ax4.set_title('Estimated Confidence Contribution by Feature Type')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig, variety_impacts, age_impacts

# --------------------
# Detailed Prediction Analysis
# --------------------
def create_detailed_prediction_analysis(model, image, variety_enc, age_norm, variety, age, label_encoder):
    """Create detailed prediction analysis"""
    
    # Prepare inputs
    img_array = tf.expand_dims(image, 0).numpy().astype(np.float32) / 255.0
    variety_array = np.array([[variety_enc]], dtype=np.int32)
    age_array = np.array([[age_norm]], dtype=np.float32)
    
    # Get prediction
    prediction = model.predict([img_array, variety_array, age_array], verbose=0)
    pred_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image with prediction
    ax1.imshow(image)
    ax1.set_title(f'Input Image\nPrediction: {label_encoder.classes_[pred_class]}\nConfidence: {confidence:.3f}', 
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Prediction probabilities
    classes = label_encoder.classes_
    y_pos = np.arange(len(classes))
    probabilities = prediction[0]
    
    bars = ax2.barh(y_pos, probabilities, color='lig htblue', alpha=0.7)
    bars[pred_class].set_color('red')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes, fontsize=10)
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim(0, 1)
    
    # Add probability values
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.3f}', va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    return fig, pred_class, confidence, probabilities

# --------------------
# Streamlit UI
# -------------------- 
def main():
    st.title("🌿 Plant Disease Classification Dashboard")
    st.markdown("""
    This interactive dashboard allows you to analyze plant disease classification using deep learning. 
    Upload an image and provide metadata to see detailed predictions and visual explanations.
    """)
    
    # Load data and model
    with st.spinner("Loading model and data..."):
        try:
            df, label_encoder, variety_encoder, model = load_data_and_model()
            st.success("✅ Model and data loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model or data: {e}")
            return
    
    # Sidebar for inputs
    st.sidebar.header("📥 Input Parameters")
    
    # Image upload
    uploaded_file = st.sidebar.file_uploader("Upload Plant Image", type=['jpg', 'jpeag', 'png'])
    
    # Variety selection
    unique_varieties = df['variety'].unique()
    variety = st.sidebar.selectbox("Select Plant Variety", unique_varieties)
    
    # Age input
    min_age = df['age'].min()
    max_age = df['age'].max()
    age = st.sidebar.slider("Plant Age (days)", min_value=int(min_age), max_value=int(max_age), 
                           value=int((min_age + max_age) / 2))
    
    # Process inputs when all are provided
    if uploaded_file is not None and variety is not None and age is not None:
        # Process image
        image = tf.io.decode_image(uploaded_file.getvalue(), channels=3)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = image.numpy().astype(np.uint8)
        
        # Encode variety and normalize age
        variety_enc = variety_encoder.transform([variety])[0]
        age_norm = (age - df['age'].min()) / (df['age'].max() - df['age'].min())
        
        # Display input information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.metric("Variety", variety)
            st.metric("Variety Encoded", variety_enc)
        with col3:
            st.metric("Age", f"{age} days")
            st.metric("Age Normalized", f"{age_norm:.3f}")
        
        st.markdown("---")
        
        # Tab interface for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction Analysis", "🔥 GradCAM", "📈 Feature Importa  nce", "ℹ️ Model Info"])
        
        with tab1:
            st.header("Detailed Prediction Analysis")
            with st.spinner("Analyzing prediction..."):
                try:
                    fig, pred_class, confidence, probabilities = create_detailed_prediction_analysis(
                        model, image, variety_enc, age_norm, variety, age, label_encoder
                    )
                    st.pyplot(fig)
                    
                    # Display prediction results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Class", label_encoder.classes_[pred_class])
                    with col2:
                        st.metric("Confidence", f"{confidence:.3f}")
                    with col3:
                        st.metric("Class Index", pred_class)
                        
                except Exception as e:
                    st.error(f"Error in prediction analysis: {e}")
        
        with tab2:
            st.header("GradCAM Visualization")
            st.markdown("""
            **GradCAM** (Gradient-weighted Class Activation Mapping) shows which regions of the image 
            were most important for the model's prediction. Red regions indicate high importance.
            """)
            
            with st.spinner("Generating GradCAM visualization..."):
                try:
                    fig, pred_class = create_gradcam_visualization(
                        model, image, variety_enc, age_norm, label_encoder
                    )
                    st.pyplot(fig)
                    st.info(f"Predicted class: **{label_encoder.classes_[pred_class]}**")
                except Exception as e:
                    st.error(f"Error generating GradCAM: {e}")
        
        with tab3:
            st.header("Feature Importance Analysis")
            st.markdown("""
            This analysis shows how different metadata features (variety and age) impact the model's prediction.
            """)
            
            with st.spinner("Analyzing feature importance..."):
                try:
                    # Get prediction for feature importance
                    img_array = tf.expand_dims(image, 0).numpy().astype(np.float32) / 255.0
                    variety_array = np.array([[variety_enc]], dtype=np.int32)
                    age_array = np.array([[age_norm]], dtype=np.float32)
                    prediction = model.predict([img_array, variety_array, age_array], verbose=0)
                    pred_class = np.argmax(prediction[0])
                    
                    fig, variety_impacts, age_impacts = create_feature_importance_analysis(
                        model, image, variety_enc, age_norm, len(variety_encoder.classes_), 
                        label_encoder, pred_class
                    )
                    st.pyplot(fig)
                    
                    # Display impact statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Variety Impact", f"{np.mean(variety_impacts):.4f}")
                    with col2:
                        st.metric("Average Age Impact", f"{np.mean(age_impacts):.4f}")
                        
                except Exception as e:
                    st.error(f"Error in feature importance analysis: {e}")
        
        with tab4:
            st.header("Model Information")   
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Class Labels")
                classes_df = pd.DataFrame({
                    'Class Index': range(len(label_encoder.classes_)),
                    'Class Name': label_encoder.classes_
                })
                st.dataframe(classes_df, use_container_width=True)
            
            with col2:
                st.subheader("🌱 Plant Varieties")
                varieties_df = pd.DataFrame({
                    'Variety Index': range(len(variety_encoder.classes_)),
                    'Variety Name': variety_encoder.classes_
                })
                st.dataframe(varieties_df, use_container_width=True)
            
            st.subheader("📊 Dataset Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Number of Classes", len(label_encoder.classes_))
            with col3:
                st.metric("Number of Varieties", len(variety_encoder.classes_))
            
            st.subheader("📈 Age Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df['age'], bins=20, alpha=0.7, color='skyblue')
            ax.set_xlabel('Age (days)')
            ax.set_ylabel('Frequency')
            ax.set_title('Age Distribution in Dataset')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
    
    else:
        # Show instructions when no input is provided
        st.info("👈 Please upload an image and provide the required metadata in the sidebar to get started.")
        
        # Show sample data
        st.subheader("📚 Sample Data from Dataset")
        st.dataframe(df[['image_id', 'label', 'variety', 'age', 'split']].head(10), use_container_width=True)
        
        # Show class distribution
        st.subheader("📊 Class Distribution")
        class_dist = df['label'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        class_dist.plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Class Distribution in Dataset')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()