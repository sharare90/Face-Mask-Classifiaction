from face_mask_classification import FaceMaskClassification


if __name__ == '__main__':
    face_mask_classification = FaceMaskClassification()
    face_mask_classification.train()
    face_mask_classification.test()
    face_mask_classification.demo()
