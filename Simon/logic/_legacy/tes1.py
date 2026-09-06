from detection_schema import Detection

d = Detection(
    cls="person",
    bbox=[100, 50, 300, 400],
    confidence=0.92,
    face_id="Rahul"
)

print(d.cls)
print(d.bbox)
print(d.face_id)
print(d.position)
