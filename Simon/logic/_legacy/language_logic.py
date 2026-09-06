# logic/language_logic.py

class LanguageLogic:
    def generate(self, detections):
        """
        detections: list of Detection objects
        returns: single string to be spoken
        """

        sentences = []

        for d in detections:
            if d.cls == "face":
                name = d.face_id if d.face_id != "Unknown" else "An unknown person"
                pos_phrase = self._get_pos_phrase(d)
                sentences.append(f"{name} is {pos_phrase}.")
            elif d.cls == "text":
                pos_phrase = self._get_pos_phrase(d)
                sentences.append(f"Text {pos_phrase}: {d.text}.")
            else:
                subject = self._get_subject(d)
                sentence = self._build_sentence(subject, d)
                sentences.append(sentence)

        return " ".join(sentences)

    def _get_subject(self, det):
        # Use name if face recognized
        if det.cls == "person" and det.face_id:
            return det.face_id
        else:
            return f"a {det.cls.replace('-', ' ')}"

    def _build_sentence(self, subject, det):
        # Distance phrase
        if det.distance == "near":
            dist_phrase = "is very close"
        elif det.distance == "medium":
            dist_phrase = "is nearby"
        else:
            dist_phrase = "is far"

        pos_phrase = self._get_pos_phrase(det)

        return f"{subject} {dist_phrase} {pos_phrase}."

    def _get_pos_phrase(self, det):
        if det.position == "front":
            return "in front of you"
        elif det.position == "left":
            return "to your left"
        elif det.position == "right":
            return "to your right"
        return "ahead"
