"""
Comprehensive Advanced AI/ML Features Demo for Aetheron Platform
Showcases all advanced modules: RL, CV, NLP, and Time Series Analysis
"""

import sys
import os
import numpy as np
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.rl.reinforcement_learning import create_rl_system
    from src.cv.computer_vision import create_cv_system, create_sample_images
    from src.nlp.natural_language_processing import create_nlp_system, create_sample_texts
    from src.timeseries.time_series_analysis import create_timeseries_system, create_sample_timeseries
except ImportError as e:
    print(f"Import error: {e}")
    print("Some advanced modules might not be available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedAIDemo:
    """Comprehensive demo of all advanced AI/ML features"""
    
    def __init__(self):
        self.results = {}
        logger.info("🚀 Initializing Advanced AI/ML Features Demo")
    
    def run_reinforcement_learning_demo(self):
        """Run reinforcement learning demonstration"""
        print("\n" + "="*60)
        print("🤖 REINFORCEMENT LEARNING DEMONSTRATION")
        print("="*60)
        
        try:
            # Create RL systems for both algorithms
            dqn_system = create_rl_system("dqn", "gridworld")
            pg_system = create_rl_system("policy_gradient", "gridworld")
            
            # Test DQN Agent
            print("\n🧠 Testing Deep Q-Network (DQN) Agent...")
            dqn_config = dqn_system['config']
            dqn_config.max_episodes = 50  # Reduced for demo
            
            dqn_experiment_id = dqn_system['tracker'].create_experiment(
                "advanced_dqn_demo",
                dqn_config,
                "Advanced DQN demonstration on GridWorld"
            )
            
            # Train DQN agent
            dqn_history = dqn_system['trainer'].train()
            dqn_evaluation = dqn_system['trainer'].evaluate(num_episodes=5)
            
            print(f"✅ DQN Training completed!")
            print(f"   Final average reward: {dqn_evaluation['mean_reward']:.2f}")
            print(f"   Reward std: {dqn_evaluation['std_reward']:.2f}")
            print(f"   Final epsilon: {dqn_system['agent'].epsilon:.3f}")
            
            # Test Policy Gradient Agent
            print("\n🎯 Testing Policy Gradient Agent...")
            pg_config = pg_system['config']
            pg_config.max_episodes = 50  # Reduced for demo
            
            pg_experiment_id = pg_system['tracker'].create_experiment(
                "advanced_pg_demo",
                pg_config,
                "Advanced Policy Gradient demonstration on GridWorld"
            )
            
            # Train Policy Gradient agent
            pg_history = pg_system['trainer'].train()
            pg_evaluation = pg_system['trainer'].evaluate(num_episodes=5)
            
            print(f"✅ Policy Gradient Training completed!")
            print(f"   Final average reward: {pg_evaluation['mean_reward']:.2f}")
            print(f"   Reward std: {pg_evaluation['std_reward']:.2f}")
            
            # Save results
            dqn_results = {
                'training_history': dqn_history,
                'evaluation_results': dqn_evaluation
            }
            dqn_system['tracker'].save_results(dqn_experiment_id, dqn_results)
            
            pg_results = {
                'training_history': pg_history,
                'evaluation_results': pg_evaluation
            }
            pg_system['tracker'].save_results(pg_experiment_id, pg_results)
            
            self.results['reinforcement_learning'] = {
                'dqn': dqn_evaluation,
                'policy_gradient': pg_evaluation,
                'status': 'success'
            }
            
            print("🎉 Reinforcement Learning Demo Completed Successfully!")
            
        except Exception as e:
            print(f"❌ RL Demo failed: {str(e)}")
            self.results['reinforcement_learning'] = {'status': 'failed', 'error': str(e)}
    
    def run_computer_vision_demo(self):
        """Run computer vision demonstration"""
        print("\n" + "="*60)
        print("👁️ COMPUTER VISION DEMONSTRATION")
        print("="*60)
        
        try:
            # Create CV system
            cv_system = create_cv_system("classification")
            
            # Create sample images and labels
            sample_images = create_sample_images(50)
            sample_labels = [i % 5 for i in range(50)]  # 5 classes
            
            print(f"📸 Created {len(sample_images)} sample images with {len(set(sample_labels))} classes")
            
            # Create experiment
            experiment_id = cv_system['tracker'].create_experiment(
                "advanced_cv_demo",
                cv_system['config'],
                "Advanced Computer Vision demonstration with CNN, detection, and segmentation"
            )
            
            # Test image processing
            print("\n🔧 Testing Image Processing...")
            processed_count = 0
            feature_samples = []
            
            for i, img in enumerate(sample_images[:10]):
                processed = cv_system['processor'].normalize_image(img)
                augmented = cv_system['processor'].augment_image(processed, "random")
                features = cv_system['processor'].extract_features(processed)
                feature_samples.append(features)
                processed_count += 1
            
            print(f"✅ Processed {processed_count} images with augmentation and feature extraction")
            
            # Test CNN classification
            print("\n🧠 Testing CNN Classification...")
            classifier = cv_system['classifier']
            
            # Training loop
            training_losses = []
            for epoch in range(20):  # Reduced for demo
                epoch_loss = 0
                for img, label in zip(sample_images[:25], sample_labels[:25]):
                    processed_img = cv_system['processor'].normalize_image(img)
                    loss = classifier.train_step(processed_img, label % 3)  # Limit to 3 classes
                    epoch_loss += loss
                
                avg_loss = epoch_loss / 25
                training_losses.append(avg_loss)
                
                if epoch % 10 == 0:
                    print(f"   Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Test predictions
            print("\n🎯 Testing Predictions...")
            predictions = []
            confidences = []
            
            for img in sample_images[25:35]:
                processed_img = cv_system['processor'].normalize_image(img)
                pred = classifier.predict(processed_img)
                prob = classifier.forward(processed_img)
                predictions.append(pred)
                confidences.append(np.max(prob))
            
            print(f"✅ Generated predictions for {len(predictions)} test images")
            print(f"   Average confidence: {np.mean(confidences):.3f}")
            
            # Test object detection
            print("\n🔍 Testing Object Detection...")
            test_image = np.random.random((128, 128))
            detections = cv_system['detector'].detect_objects(test_image, threshold=0.3)
            filtered_detections = cv_system['detector'].non_max_suppression(detections)
            
            print(f"✅ Object Detection completed")
            print(f"   Raw detections: {len(detections)}")
            print(f"   After NMS: {len(filtered_detections)}")
            
            # Test segmentation
            print("\n🗺️ Testing Image Segmentation...")
            test_image_seg = sample_images[0]
            kmeans_result = cv_system['segmentation'].kmeans_clustering(test_image_seg)
            edge_result = cv_system['segmentation'].edge_based_segmentation(test_image_seg)
            
            unique_segments = len(np.unique(kmeans_result))
            edge_pixels = np.sum(edge_result)
            
            print(f"✅ Segmentation completed")
            print(f"   K-means segments: {unique_segments}")
            print(f"   Edge pixels detected: {edge_pixels}")
            
            # Save results
            results = {
                'classification': {
                    'training_losses': training_losses,
                    'final_loss': training_losses[-1] if training_losses else 0,
                    'num_predictions': len(predictions),
                    'avg_confidence': float(np.mean(confidences))
                },
                'detection': {
                    'raw_detections': len(detections),
                    'filtered_detections': len(filtered_detections)
                },
                'segmentation': {
                    'kmeans_segments': int(unique_segments),
                    'edge_pixels': int(edge_pixels)
                },
                'processing': {
                    'processed_images': processed_count,
                    'feature_samples': feature_samples[:3]  # Save first 3 samples
                }
            }
            
            cv_system['tracker'].save_results(experiment_id, results)
            
            self.results['computer_vision'] = {
                **results,
                'status': 'success'
            }
            
            print("🎉 Computer Vision Demo Completed Successfully!")
            
        except Exception as e:
            print(f"❌ CV Demo failed: {str(e)}")
            self.results['computer_vision'] = {'status': 'failed', 'error': str(e)}
    
    def run_nlp_demo(self):
        """Run natural language processing demonstration"""
        print("\n" + "="*60)
        print("💬 NATURAL LANGUAGE PROCESSING DEMONSTRATION")
        print("="*60)
        
        try:
            # Create NLP system
            nlp_system = create_nlp_system("sentiment")
            
            # Get sample data
            sample_texts, sample_labels = create_sample_texts()
            
            print(f"📝 Loaded {len(sample_texts)} sample texts for analysis")
            
            # Create experiment
            experiment_id = nlp_system['tracker'].create_experiment(
                "advanced_nlp_demo",
                nlp_system['config'],
                "Advanced NLP demonstration with sentiment analysis, classification, and summarization"
            )
            
            # Build vocabulary
            print("\n📚 Building Vocabulary...")
            nlp_system['vocabulary'].build_vocabulary(sample_texts, nlp_system['preprocessor'])
            vocab_size = len(nlp_system['vocabulary'].word_to_id)
            print(f"✅ Built vocabulary with {vocab_size} words")
            
            # Test text preprocessing
            print("\n🔧 Testing Text Preprocessing...")
            preprocessing_results = []
            
            for i, text in enumerate(sample_texts[:5]):
                tokens = nlp_system['preprocessor'].tokenize(text)
                features = nlp_system['preprocessor'].extract_features(text)
                sequence = nlp_system['vocabulary'].text_to_sequence(
                    text, nlp_system['preprocessor'], 20
                )
                
                preprocessing_results.append({
                    'text': text,
                    'num_tokens': len(tokens),
                    'sequence_length': len(sequence),
                    'features': features
                })
                
                print(f"   Text {i+1}: {len(tokens)} tokens, {len(sequence)} sequence length")
            
            # Test sentiment analysis
            print("\n😊 Testing Sentiment Analysis...")
            sentiment_analyzer = nlp_system['sentiment_analyzer']
            
            # Training loop
            sentiment_losses = []
            for epoch in range(15):  # Reduced for demo
                epoch_loss = 0
                for text, label in zip(sample_texts, sample_labels):
                    sequence = nlp_system['vocabulary'].text_to_sequence(
                        text, nlp_system['preprocessor'], 20
                    )
                    loss = sentiment_analyzer.train_step(sequence, label)
                    epoch_loss += loss
                
                avg_loss = epoch_loss / len(sample_texts)
                sentiment_losses.append(avg_loss)
                
                if epoch % 5 == 0:
                    print(f"   Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Test sentiment predictions
            print("\n🎯 Testing Sentiment Predictions...")
            sentiment_predictions = []
            
            for i, text in enumerate(sample_texts[:8]):
                sequence = nlp_system['vocabulary'].text_to_sequence(
                    text, nlp_system['preprocessor'], 20
                )
                result = sentiment_analyzer.predict_sentiment(sequence)
                sentiment_predictions.append(result)
                
                true_sentiment = ['negative', 'neutral', 'positive'][sample_labels[i]]
                print(f"   Text: '{text[:50]}...'")
                print(f"   Predicted: {result['sentiment']} (conf: {result['confidence']:.3f})")
                print(f"   True: {true_sentiment}")
                print()
            
            # Test text classification
            print("\n📊 Testing Text Classification...")
            classifier = nlp_system['text_classifier']
            
            # Training loop
            classification_losses = []
            for epoch in range(10):  # Reduced for demo
                epoch_loss = 0
                for text, label in zip(sample_texts, sample_labels):
                    sequence = nlp_system['vocabulary'].text_to_sequence(
                        text, nlp_system['preprocessor'], 20
                    )
                    loss = classifier.train_step(sequence, label)
                    epoch_loss += loss
                
                avg_loss = epoch_loss / len(sample_texts)
                classification_losses.append(avg_loss)
                
                if epoch % 5 == 0:
                    print(f"   Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Test summarization
            print("\n📄 Testing Text Summarization...")
            long_text = " ".join(sample_texts)
            summary = nlp_system['summarizer'].extractive_summarize(
                long_text, nlp_system['preprocessor']
            )
            
            compression_ratio = len(summary) / len(long_text)
            print(f"✅ Summarization completed")
            print(f"   Original length: {len(long_text)} characters")
            print(f"   Summary length: {len(summary)} characters")
            print(f"   Compression ratio: {compression_ratio:.2f}")
            print(f"   Summary: {summary[:200]}...")
            
            # Calculate accuracy
            correct_predictions = 0
            for i, pred in enumerate(sentiment_predictions):
                true_label = ['negative', 'neutral', 'positive'][sample_labels[i]]
                if pred['sentiment'] == true_label:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(sentiment_predictions) if sentiment_predictions else 0
            
            # Save results
            results = {
                'vocabulary': {
                    'size': vocab_size,
                    'max_vocab_size': nlp_system['config'].vocab_size
                },
                'sentiment_analysis': {
                    'training_losses': sentiment_losses,
                    'final_loss': sentiment_losses[-1] if sentiment_losses else 0,
                    'accuracy': accuracy,
                    'num_predictions': len(sentiment_predictions)
                },
                'classification': {
                    'training_losses': classification_losses,
                    'final_loss': classification_losses[-1] if classification_losses else 0
                },
                'summarization': {
                    'compression_ratio': compression_ratio,
                    'original_length': len(long_text),
                    'summary_length': len(summary)
                },
                'preprocessing': {
                    'num_texts_processed': len(preprocessing_results),
                    'sample_features': preprocessing_results[0]['features'] if preprocessing_results else {}
                }
            }
            
            nlp_system['tracker'].save_results(experiment_id, results)
            
            self.results['natural_language_processing'] = {
                **results,
                'status': 'success'
            }
            
            print("🎉 NLP Demo Completed Successfully!")
            
        except Exception as e:
            print(f"❌ NLP Demo failed: {str(e)}")
            self.results['natural_language_processing'] = {'status': 'failed', 'error': str(e)}
    
    def run_timeseries_demo(self):
        """Run time series analysis demonstration"""
        print("\n" + "="*60)
        print("📈 TIME SERIES ANALYSIS DEMONSTRATION")
        print("="*60)
        
        try:
            # Create time series system
            ts_system = create_timeseries_system("forecasting")
            
            # Create sample data
            sample_series = create_sample_timeseries(length=400, noise_level=0.15)
            
            print(f"📊 Generated time series with {len(sample_series)} data points")
            
            # Create experiment
            experiment_id = ts_system['tracker'].create_experiment(
                "advanced_timeseries_demo",
                ts_system['config'],
                "Advanced Time Series Analysis with forecasting, anomaly detection, and decomposition"
            )
            
            # Split data
            train_size = int(0.8 * len(sample_series))
            train_series = sample_series[:train_size]
            test_series = sample_series[train_size:]
            
            print(f"📈 Training on {len(train_series)} points, testing on {len(test_series)} points")
            
            # Fit forecaster
            print("\n🔧 Fitting Forecasting Model...")
            decomposition = ts_system['forecaster'].fit(train_series)
            
            trend_strength = decomposition['trend']['trend_strength']
            seasonal_strength = decomposition['seasonal']['seasonal_strength']
            has_seasonality = decomposition['seasonal']['has_seasonality']
            
            print(f"✅ Model fitting completed")
            print(f"   Trend strength: {trend_strength:.3f}")
            print(f"   Seasonal strength: {seasonal_strength:.3f}")
            print(f"   Has seasonality: {has_seasonality}")
            
            # Generate forecasts
            print("\n🔮 Generating Forecasts...")
            forecast_steps = min(len(test_series), 20)
            forecast_results = ts_system['forecaster'].forecast(forecast_steps)
            
            print(f"✅ Generated {forecast_steps} forecast points")
            
            # Calculate forecast accuracy
            actual_values = test_series[:forecast_steps]
            predicted_values = forecast_results['forecast'][:forecast_steps]
            
            mae = np.mean(np.abs(actual_values - predicted_values))
            mse = np.mean((actual_values - predicted_values) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_values - predicted_values) / (actual_values + 1e-8))) * 100
            
            print(f"📊 Forecast Accuracy Metrics:")
            print(f"   MAE (Mean Absolute Error): {mae:.3f}")
            print(f"   RMSE (Root Mean Square Error): {rmse:.3f}")
            print(f"   MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
            
            # Test anomaly detection
            print("\n🚨 Testing Anomaly Detection...")
            
            # Statistical method
            from src.timeseries.time_series_analysis import AnomalyDetector
            anomaly_detector_stat = AnomalyDetector(method="statistical")
            anomalies_stat = anomaly_detector_stat.detect_anomalies(train_series, threshold=2.5)
            
            # Moving window method
            anomaly_detector_window = AnomalyDetector(method="moving_window", window_size=20)
            anomalies_window = anomaly_detector_window.detect_anomalies(train_series, threshold=2.0)
            
            print(f"✅ Anomaly Detection completed")
            print(f"   Statistical method: {np.sum(anomalies_stat)} anomalies ({100*np.sum(anomalies_stat)/len(anomalies_stat):.1f}%)")
            print(f"   Moving window method: {np.sum(anomalies_window)} anomalies ({100*np.sum(anomalies_window)/len(anomalies_window):.1f}%)")
            
            # Test outlier detection
            print("\n📊 Testing Outlier Detection...")
            from src.timeseries.time_series_analysis import TimeSeriesPreprocessor
            preprocessor = TimeSeriesPreprocessor()
            
            outliers_iqr = preprocessor.detect_outliers(train_series, method="iqr", threshold=1.5)
            outliers_zscore = preprocessor.detect_outliers(train_series, method="zscore", threshold=2.5)
            
            print(f"✅ Outlier Detection completed")
            print(f"   IQR method: {np.sum(outliers_iqr)} outliers ({100*np.sum(outliers_iqr)/len(outliers_iqr):.1f}%)")
            print(f"   Z-score method: {np.sum(outliers_zscore)} outliers ({100*np.sum(outliers_zscore)/len(outliers_zscore):.1f}%)")
            
            # Test trend analysis
            print("\n📈 Testing Trend Analysis...")
            from src.timeseries.time_series_analysis import TrendAnalyzer
            trend_analyzer = TrendAnalyzer("linear")
            trend_results = trend_analyzer.fit_trend(train_series)
            
            print(f"✅ Trend Analysis completed")
            print(f"   Trend coefficients: {trend_results['coefficients']}")
            print(f"   Trend strength: {trend_results['trend_strength']:.3f}")
            
            # Test seasonality detection
            print("\n🔄 Testing Seasonality Detection...")
            from src.timeseries.time_series_analysis import SeasonalityDetector
            seasonality_detector = SeasonalityDetector(seasonal_period=24)
            seasonality_results = seasonality_detector.detect_seasonality(trend_results['detrended'])
            
            print(f"✅ Seasonality Detection completed")
            print(f"   Seasonal strength: {seasonality_results['seasonal_strength']:.3f}")
            print(f"   Has seasonality: {seasonality_results['has_seasonality']}")
            
            # Save results
            results = {
                'forecast_accuracy': {
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mape': float(mape)
                },
                'decomposition': {
                    'trend_strength': float(trend_strength),
                    'seasonal_strength': float(seasonal_strength),
                    'has_seasonality': bool(has_seasonality)
                },
                'anomaly_detection': {
                    'statistical': {
                        'num_anomalies': int(np.sum(anomalies_stat)),
                        'anomaly_rate': float(np.sum(anomalies_stat) / len(anomalies_stat))
                    },
                    'moving_window': {
                        'num_anomalies': int(np.sum(anomalies_window)),
                        'anomaly_rate': float(np.sum(anomalies_window) / len(anomalies_window))
                    }
                },
                'outlier_detection': {
                    'iqr': {
                        'num_outliers': int(np.sum(outliers_iqr)),
                        'outlier_rate': float(np.sum(outliers_iqr) / len(outliers_iqr))
                    },
                    'zscore': {
                        'num_outliers': int(np.sum(outliers_zscore)),
                        'outlier_rate': float(np.sum(outliers_zscore) / len(outliers_zscore))
                    }
                },
                'trend_analysis': trend_results,
                'seasonality_analysis': {
                    'seasonal_strength': float(seasonality_results['seasonal_strength']),
                    'has_seasonality': bool(seasonality_results['has_seasonality'])
                },
                'data_info': {
                    'total_points': len(sample_series),
                    'train_points': len(train_series),
                    'test_points': len(test_series),
                    'forecast_points': forecast_steps
                }
            }
            
            ts_system['tracker'].save_results(experiment_id, results)
            
            self.results['time_series_analysis'] = {
                **results,
                'status': 'success'
            }
            
            print("🎉 Time Series Analysis Demo Completed Successfully!")
            
        except Exception as e:
            print(f"❌ Time Series Demo failed: {str(e)}")
            self.results['time_series_analysis'] = {'status': 'failed', 'error': str(e)}
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE DEMO SUMMARY REPORT")
        print("="*60)
        
        total_modules = 4
        successful_modules = sum(1 for result in self.results.values() 
                               if result.get('status') == 'success')
        
        print(f"\n🎯 Overall Success Rate: {successful_modules}/{total_modules} ({100*successful_modules/total_modules:.1f}%)")
        
        for module_name, result in self.results.items():
            status = result.get('status', 'unknown')
            status_icon = "✅" if status == 'success' else "❌"
            
            print(f"\n{status_icon} {module_name.replace('_', ' ').title()}:")
            
            if status == 'success':
                if module_name == 'reinforcement_learning':
                    dqn_reward = result.get('dqn', {}).get('mean_reward', 0)
                    pg_reward = result.get('policy_gradient', {}).get('mean_reward', 0)
                    print(f"   DQN Mean Reward: {dqn_reward:.2f}")
                    print(f"   Policy Gradient Mean Reward: {pg_reward:.2f}")
                
                elif module_name == 'computer_vision':
                    final_loss = result.get('classification', {}).get('final_loss', 0)
                    detections = result.get('detection', {}).get('filtered_detections', 0)
                    segments = result.get('segmentation', {}).get('kmeans_segments', 0)
                    print(f"   Classification Loss: {final_loss:.4f}")
                    print(f"   Object Detections: {detections}")
                    print(f"   Image Segments: {segments}")
                
                elif module_name == 'natural_language_processing':
                    accuracy = result.get('sentiment_analysis', {}).get('accuracy', 0)
                    vocab_size = result.get('vocabulary', {}).get('size', 0)
                    compression = result.get('summarization', {}).get('compression_ratio', 0)
                    print(f"   Sentiment Accuracy: {accuracy:.2f}")
                    print(f"   Vocabulary Size: {vocab_size}")
                    print(f"   Text Compression: {compression:.2f}")
                
                elif module_name == 'time_series_analysis':
                    mae = result.get('forecast_accuracy', {}).get('mae', 0)
                    mape = result.get('forecast_accuracy', {}).get('mape', 0)
                    seasonal = result.get('decomposition', {}).get('has_seasonality', False)
                    print(f"   Forecast MAE: {mae:.3f}")
                    print(f"   Forecast MAPE: {mape:.2f}%")
                    print(f"   Seasonality Detected: {seasonal}")
            
            else:
                error = result.get('error', 'Unknown error')
                print(f"   Error: {error}")
        
        # Performance Summary
        print("\n📊 Performance Highlights:")
        if self.results.get('reinforcement_learning', {}).get('status') == 'success':
            print("   • Reinforcement Learning agents successfully trained on GridWorld")
        if self.results.get('computer_vision', {}).get('status') == 'success':
            print("   • Computer Vision pipeline with CNN, detection, and segmentation")
        if self.results.get('natural_language_processing', {}).get('status') == 'success':
            print("   • NLP system with sentiment analysis and text summarization")
        if self.results.get('time_series_analysis', {}).get('status') == 'success':
            print("   • Time Series analysis with forecasting and anomaly detection")
        
        print("\n🎉 Advanced AI/ML Features Demo Completed!")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Total Execution Time: See individual module logs")
        
        return self.results

def main():
    """Main demo execution"""
    print("🚀 AETHERON ADVANCED AI/ML FEATURES DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases cutting-edge AI/ML capabilities:")
    print("• Reinforcement Learning (DQN & Policy Gradient)")
    print("• Computer Vision (CNN, Detection, Segmentation)")
    print("• Natural Language Processing (Sentiment, Summarization)")
    print("• Time Series Analysis (Forecasting, Anomaly Detection)")
    print("=" * 60)
    
    # Initialize demo
    demo = AdvancedAIDemo()
    
    try:
        # Run all demonstrations
        demo.run_reinforcement_learning_demo()
        demo.run_computer_vision_demo()
        demo.run_nlp_demo()
        demo.run_timeseries_demo()
        
        # Generate final report
        final_results = demo.generate_summary_report()
        
        # Save consolidated results
        os.makedirs('reports/advanced_demo', exist_ok=True)
        
        import json
        with open('reports/advanced_demo/comprehensive_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: reports/advanced_demo/comprehensive_results.json")
        
        return final_results
        
    except Exception as e:
        print(f"\n❌ Demo execution failed: {str(e)}")
        logger.error(f"Demo execution failed: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    results = main()
    if results:
        successful_count = sum(1 for r in results.values() if r.get('status') == 'success')
        print(f"\n🎯 Final Status: {successful_count}/4 modules completed successfully")
    else:
        print("\n❌ Demo failed to complete")
