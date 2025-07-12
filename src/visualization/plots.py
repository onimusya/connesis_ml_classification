import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.tree import export_text
from src.utils.logger import handle_error, log_info


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Create confusion matrix heatmap"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, 
                        text_auto=True, 
                        aspect="auto",
                        x=class_names,
                        y=class_names,
                        labels=dict(x="Predicted", y="Actual"),
                        title="Confusion Matrix",
                        color_continuous_scale='Blues')
        
        log_info("Confusion matrix plot created")
        return fig
        
    except Exception as e:
        handle_error(e, "Error creating confusion matrix plot")
        return None


def plot_feature_importance(model, feature_names):
    """Plot feature importance if available"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(importance_df, 
                         x='importance', 
                         y='feature',
                         title="Feature Importance",
                         orientation='h')
            
            log_info("Feature importance plot created")
            return fig
        else:
            log_info("Model does not provide feature importance")
            return None
            
    except Exception as e:
        handle_error(e, "Error creating feature importance plot")
        return None


def plot_roc_curve(y_true, y_prob, class_names):
    """Plot ROC curve"""
    try:
        fig = go.Figure()
        
        if len(class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, 
                                    name=f'ROC Curve (AUC = {roc_auc:.2f})'))
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, 
                                        name=f'Class {class_names[i]} (AUC = {roc_auc:.2f})'))
        
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                mode='lines', 
                                line=dict(dash='dash'),
                                name='Random'))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        log_info("ROC curve plot created")
        return fig
        
    except Exception as e:
        handle_error(e, "Error creating ROC curve plot")
        return None


def plot_correlation_heatmap(df, feature_cols):
    """Create correlation heatmap for all features (numerical and encoded categorical)"""
    try:
        # Create a copy of the feature data
        corr_data = df[feature_cols].copy()
        
        # Convert categorical columns to numerical using label encoding
        label_encoders = {}
        for col in corr_data.columns:
            if corr_data[col].dtype == 'object':
                le = LabelEncoder()
                # Handle NaN values
                mask = corr_data[col].notna()
                if mask.sum() > 0:
                    corr_data.loc[mask, col] = le.fit_transform(corr_data.loc[mask, col].astype(str))
                    label_encoders[col] = le
                else:
                    corr_data[col] = 0  # Fill with 0 if all NaN
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        # Create heatmap
        fig = px.imshow(corr_matrix,
                        text_auto='.2f',
                        aspect="auto",
                        title="Feature Correlation Heatmap",
                        color_continuous_scale='RdBu_r',
                        zmin=-1,
                        zmax=1)
        
        fig.update_layout(
            width=800,
            height=600
        )
        
        log_info("Correlation heatmap created")
        return fig
        
    except Exception as e:
        handle_error(e, "Error creating correlation heatmap")
        return None


def get_model_specific_view(model, model_name, feature_names):
    """Get model-specific visualization or information"""
    try:
        if model_name == "Decision Tree":
            tree_rules = export_text(model, feature_names=feature_names)
            log_info("Decision tree structure extracted")
            return tree_rules
        else:
            log_info(f"No specific visualization available for {model_name}")
            return None
            
    except Exception as e:
        handle_error(e, f"Error getting {model_name} specific view")
        return None