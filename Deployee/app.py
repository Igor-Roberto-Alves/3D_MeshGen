from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)


app.config['UPLOAD_FOLDER'] = 'generated_clouds'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate a point cloud from class and latent vectors.
    
    Request JSON:
    {
        "class": "chair",
        "latent": {"z1": 0.5, "z2": -0.2, "z3": 0.1, "z4": 0.0}
    }
    
    Response JSON:
    {
        "success": true,
        "class": "chair",
        "points_count": 5000,
        "latent": {...},
        "timestamp": "2024-01-15T10:30:45",
        "message": "Point cloud generated successfully"
    }
    """
    try:
        data = request.get_json()
        class_name = data.get('class', 'chair').lower()
        latent = data.get('latent', {})
        
        # Validate class
        if class_name not in model.classes:
            return jsonify({
                'success': False,
                'error': f'Invalid class. Choose from: {model.classes}'
            }), 400
        
        # Validate latent vectors
        valid_keys = ['z1', 'z2', 'z3', 'z4']
        for key in valid_keys:
            if key not in latent:
                latent[key] = 0.0
            else:
                latent[key] = float(latent[key])
                # Clamp to [-1, 1]
                latent[key] = max(-1.0, min(1.0, latent[key]))
        
        # Generate point cloud
        point_cloud = model.generate(class_name, latent)
        
        # Save point cloud as .npy file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{class_name}_{timestamp}.npy"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        np.save(filepath, point_cloud)
        
        # Prepare response
        response = {
            'success': True,
            'class': class_name,
            'points_count': len(point_cloud),
            'latent': latent,
            'timestamp': datetime.now().isoformat(),
            'message': 'Point cloud generated successfully',
            'filename': filename
        }
        
        print(f"[✓] Generated {class_name}: {len(point_cloud)} points")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"[✗] Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    """Download generated point cloud file"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/list', methods=['GET'])
def list_generations():
    """List all generated point clouds"""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        file_info = []
        for f in sorted(files, reverse=True):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f)
            size = os.path.getsize(filepath)
            file_info.append({
                'filename': f,
                'size_bytes': size,
                'size_mb': round(size / (1024*1024), 2)
            })
        return jsonify({
            'success': True,
            'count': len(file_info),
            'files': file_info
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Get model and configuration info"""
    return jsonify({
        'model': 'Diffuse-VAE Cloud Points Generator',
        'latent_dim': model.latent_dim,
        'classes': model.classes,
        'points_per_cloud': model.points_per_cloud,
        'version': '1.0.0-alpha',
        'status': 'ready'
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Diffuse-VAE Cloud Points Server")
    print("=" * 60)
    print("🚀 Server running on http://localhost:5000")
    print("📱 Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)