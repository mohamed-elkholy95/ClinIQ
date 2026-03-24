"""Database seeding script for initial ICD-10 codes and demo data."""

import asyncio
import logging
from uuid import uuid4

from sqlalchemy import select

from app.core.security import get_password_hash
from app.db.models import ICDCode, User
from app.db.session import get_db_context

logger = logging.getLogger(__name__)

# Common ICD-10 codes for seeding
ICD10_SEED_DATA = [
    ("A09", "Infectious gastroenteritis and colitis, unspecified", "Certain infectious and parasitic diseases", "Intestinal infectious diseases"),
    ("B34.9", "Viral infection, unspecified", "Certain infectious and parasitic diseases", "Other viral diseases"),
    ("C50.9", "Malignant neoplasm of breast, unspecified", "Neoplasms", "Malignant neoplasm of breast"),
    ("D64.9", "Anemia, unspecified", "Diseases of the blood", "Other anemias"),
    ("E11.9", "Type 2 diabetes mellitus without complications", "Endocrine diseases", "Diabetes mellitus"),
    ("E11.65", "Type 2 diabetes mellitus with hyperglycemia", "Endocrine diseases", "Diabetes mellitus"),
    ("E78.5", "Dyslipidemia, unspecified", "Endocrine diseases", "Disorders of lipoprotein metabolism"),
    ("F32.9", "Major depressive disorder, single episode, unspecified", "Mental disorders", "Depressive episodes"),
    ("F41.1", "Generalized anxiety disorder", "Mental disorders", "Other anxiety disorders"),
    ("G43.9", "Migraine, unspecified", "Diseases of the nervous system", "Migraine"),
    ("G47.0", "Insomnia", "Diseases of the nervous system", "Sleep disorders"),
    ("I10", "Essential (primary) hypertension", "Diseases of the circulatory system", "Hypertensive diseases"),
    ("I25.10", "Atherosclerotic heart disease of native coronary artery", "Diseases of the circulatory system", "Ischemic heart diseases"),
    ("I48.91", "Unspecified atrial fibrillation", "Diseases of the circulatory system", "Atrial fibrillation"),
    ("I50.9", "Heart failure, unspecified", "Diseases of the circulatory system", "Heart failure"),
    ("J06.9", "Acute upper respiratory infection, unspecified", "Diseases of the respiratory system", "Acute upper respiratory infections"),
    ("J18.9", "Pneumonia, unspecified organism", "Diseases of the respiratory system", "Pneumonia"),
    ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation", "Diseases of the respiratory system", "COPD"),
    ("J45.9", "Asthma, unspecified", "Diseases of the respiratory system", "Asthma"),
    ("K21.0", "Gastro-esophageal reflux disease with esophagitis", "Diseases of the digestive system", "GERD"),
    ("K35.80", "Unspecified acute appendicitis", "Diseases of the digestive system", "Acute appendicitis"),
    ("K59.0", "Constipation", "Diseases of the digestive system", "Functional intestinal disorders"),
    ("K76.0", "Fatty (change of) liver, not elsewhere classified", "Diseases of the digestive system", "Other diseases of liver"),
    ("M54.5", "Low back pain", "Diseases of the musculoskeletal system", "Dorsalgia"),
    ("M79.3", "Panniculitis, unspecified", "Diseases of the musculoskeletal system", "Other soft tissue disorders"),
    ("N18.9", "Chronic kidney disease, unspecified", "Diseases of the genitourinary system", "Chronic kidney disease"),
    ("N39.0", "Urinary tract infection, site not specified", "Diseases of the genitourinary system", "Other disorders of urinary system"),
    ("R05", "Cough", "Symptoms and signs", "Symptoms involving the respiratory system"),
    ("R10.9", "Unspecified abdominal pain", "Symptoms and signs", "Abdominal and pelvic pain"),
    ("R51", "Headache", "Symptoms and signs", "Pain"),
    ("Z00.00", "Encounter for general adult medical examination", "Factors influencing health status", "General examination"),
    ("Z23", "Encounter for immunization", "Factors influencing health status", "Immunization"),
    # Dental-specific ICD-10 codes
    ("K02.9", "Dental caries, unspecified", "Diseases of the digestive system", "Dental caries"),
    ("K04.0", "Pulpitis", "Diseases of the digestive system", "Diseases of pulp and periapical tissues"),
    ("K05.1", "Chronic gingivitis", "Diseases of the digestive system", "Gingivitis and periodontal diseases"),
    ("K05.3", "Chronic periodontitis", "Diseases of the digestive system", "Gingivitis and periodontal diseases"),
    ("K08.1", "Complete loss of teeth", "Diseases of the digestive system", "Other disorders of teeth"),
    ("K12.0", "Recurrent oral aphthae", "Diseases of the digestive system", "Stomatitis and related lesions"),
]


async def seed_icd_codes() -> int:
    """Seed ICD-10 reference codes."""
    count = 0
    async with get_db_context() as db:
        for code, description, chapter, category in ICD10_SEED_DATA:
            existing = await db.execute(select(ICDCode).where(ICDCode.code == code))
            if existing.scalar_one_or_none() is None:
                db.add(ICDCode(
                    code=code,
                    description=description,
                    chapter=chapter,
                    category=category,
                    is_active=True,
                ))
                count += 1
    logger.info("Seeded %d ICD-10 codes", count)
    return count


async def seed_admin_user() -> None:
    """Create default admin user if none exists."""
    async with get_db_context() as db:
        existing = await db.execute(select(User).where(User.email == "admin@cliniq.local"))
        if existing.scalar_one_or_none() is None:
            db.add(User(
                id=uuid4(),
                email="admin@cliniq.local",
                hashed_password=get_password_hash("changeme"),
                full_name="ClinIQ Admin",
                is_active=True,
                is_superuser=True,
                role="admin",
            ))
            logger.info("Created default admin user (admin@cliniq.local)")


async def seed_all() -> None:
    """Run all seed operations."""
    await seed_icd_codes()
    await seed_admin_user()
    logger.info("Database seeding complete")


def main() -> None:
    """Entry point for `python -m app.db.seed`."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed_all())


if __name__ == "__main__":
    main()
