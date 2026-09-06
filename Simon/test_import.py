try:
    import vision.pipeline.vision_pipeline
    print('OK')
except Exception as e:
    import traceback
    traceback.print_exc()
