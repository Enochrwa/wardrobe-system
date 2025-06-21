from fastapi import APIRouter,Path, Depends, HTTPException, status, File, UploadFile
from typing import List, Optional, Dict, Any
from datetime import datetime
import shutil
import uuid
import os
import logging
from PIL import Image
import io 
from sqlalchemy.orm import Session
from ..services.outfit_recommendation_engine import get_recommendation_engine 
from ..schemas import ml_features as schemas_ml # Corrected import for ml_features schemas
  
from .. import tables as schemas
from .. import model as models
from ..services import ai_embedding
from ..services.color_detector import get_color_detector
from ..services.clothing_classifier import get_clothing_classifier
from ..services import user_style_profile_service # Import the new service
from ..security import get_current_user
from ..db.database import get_db

router = APIRouter(
    prefix="/wardrobe",
    tags=["Wardrobe"],
    responses={404: {"description": "Not found"}},
)

STATIC_DIR = "static"
WARDROBE_IMAGES_DIR = os.path.join(STATIC_DIR, "wardrobe_images")
os.makedirs(WARDROBE_IMAGES_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"]

# Helper function for textual style feature extraction
def extract_textual_style_features(name: Optional[str], category: Optional[str], tags: Optional[List[str]]) -> Dict[str, Any]:
    """
    Extracts basic style features from textual metadata of a wardrobe item.
    """
    keywords = set()
    if name:
        # Split name into words, convert to lowercase
        keywords.update(word.lower() for word in name.split())
    if category:
        keywords.add(category.lower())
    if tags:
        keywords.update(tag.lower() for tag in tags)

    defined_styles = {
        "casual": ["casual", "relaxed", "everyday", "comfy", "jeans", "t-shirt", "sneakers"],
        "formal": ["formal", "elegant", "dressy", "sophisticated", "suit", "gown", "blazer", "tie", "dress shoes"],
        "sporty": ["sporty", "athletic", "active", "gym", "sport", "tracksuit", "trainers"],
        "bohemian": ["boho", "bohemian", "hippie", "flowy", "fringe"],
        "vintage": ["vintage", "retro", "antique"],
        "minimalist": ["minimalist", "simple", "clean", "basic", "unembellished"],
        "streetwear": ["streetwear", "urban", "hip-hop", "graphic tee", "hoodie"],
        "classic": ["classic", "timeless", "traditional", "iconic"],
        "business": ["business", "professional", "work", "office", "corporate", "smart"],
        "party": ["party", "festive", "evening", "cocktail", "clubbing", "sequin"],
        "outerwear": ["outerwear", "coat", "jacket", "parka", "cardigan"], # Added outerwear as a style
        "summer": ["summer", "beach", "sundress", "shorts", "sandals"],
        "winter": ["winter", "warm", "cozy", "knit", "wool", "scarf"]
    }

    identified_styles = []
    for style, style_keys in defined_styles.items():
        if any(key in keywords for key in style_keys) or any(key in ' '.join(keywords) for key in style_keys if ' ' in key): # Check for multi-word keys
            identified_styles.append(style)
    
    all_keywords = list(keywords)
    if tags: # Ensure tags are added if not already from keywords set
        all_keywords.extend([tag.lower() for tag in tags if tag.lower() not in keywords])
    
    stopwords = {"a", "an", "the", "is", "are", "and", "of", "for", "with", "on", "in", "to", "it", "item"}
    meaningful_keywords = sorted(list(set(kw for kw in all_keywords if kw not in stopwords and len(kw) > 2)))

    return {
        "identified_styles": sorted(list(set(identified_styles))),
        "raw_keywords": meaningful_keywords
    }


@router.post("/items/", response_model=schemas.WardrobeItem, status_code=status.HTTP_201_CREATED)
async def create_wardrobe_item(
    item: schemas.WardrobeItemCreate = Depends(),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    item_data = item.model_dump()
    
    # Initialize all AI and color fields to None or default
    item_data.update({
        'ai_embedding': None, 'ai_dominant_colors': None, 'dominant_color_rgb': None,
        'dominant_color_hex': None, 'dominant_color_name': None, 'color_palette': None,
        'color_properties': None, 'ai_classification': None, 'style_features': None
    })

    if image and image.filename:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image type. Allowed types: {ALLOWED_CONTENT_TYPES}")

        # Read image bytes once for all services
        image_bytes_content = await image.read()
        await image.seek(0) # Reset cursor for saving the file

        if len(image_bytes_content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Image too large. Max size: {MAX_FILE_SIZE_BYTES // (1024*1024)}MB")

        unique_id = uuid.uuid4()
        extension = os.path.splitext(image.filename)[1]
        filename = f"{unique_id}{extension}"
        file_path = os.path.join(WARDROBE_IMAGES_DIR, filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            item_data['image_url'] = f"/{file_path}"

            # AI Processing using the single read image_bytes_content
            pil_image_for_embedding = Image.open(io.BytesIO(image_bytes_content)) # For embedding service if it needs PIL

            # 1. Embedding
            try:
                item_data['ai_embedding'] = ai_embedding.get_image_embedding(pil_image_for_embedding)
            except Exception as e:
                logger.error(f"Error generating image embedding: {e}")

            # 2. Color Detection (using ColorDetector)
            try:
                color_detector = get_color_detector()
                dominant_color_info = color_detector.extract_dominant_color(image_bytes_content)
                palette_info = color_detector.extract_color_palette(image_bytes_content)

                if dominant_color_info.get("success"):
                    item_data['dominant_color_rgb'] = dominant_color_info["dominant_color"]["rgb"]
                    item_data['dominant_color_hex'] = dominant_color_info["dominant_color"]["hex"]
                    item_data['dominant_color_name'] = dominant_color_info["dominant_color"]["name"]
                    item_data['color_properties'] = dominant_color_info["properties"]
                
                if palette_info.get("success"):
                    item_data['color_palette'] = palette_info["palette"]
                    # Populate ai_dominant_colors from the new palette for compatibility or if needed
                    item_data['ai_dominant_colors'] = [p_color['hex'] for p_color in palette_info["palette"]] if palette_info["palette"] else []
                else: # Fallback for ai_dominant_colors if full palette extraction fails
                    item_data['ai_dominant_colors'] = [dominant_color_info["dominant_color"]["hex"]] if dominant_color_info.get("success") else []


            except Exception as e:
                logger.error(f"Error processing colors with ColorDetector: {e}")

            # 3. Clothing Classification
            try:
                classifier = get_clothing_classifier()
                classification_result = classifier.classify_clothing_item(image_bytes_content)
                if classification_result.get("success"):
                    item_data['ai_classification'] = {
                        "category": classification_result["clothing_type"],
                        "confidence": classification_result["confidence"],
                        "details": classification_result["detailed_predictions"]
                    }
            except Exception as e:
                logger.error(f"Error classifying clothing: {e}")
        
        except Exception as e:
            logger.error(f"Error saving image or during AI processing: {e}")
            item_data['image_url'] = item.image_url # Fallback
        finally:
            image.file.close()
            
    elif item.image_url:
        item_data['image_url'] = item.image_url
    else:
        item_data['image_url'] = None

    # 4. Style Features (Textual) - This is done regardless of image, based on item_data
    try:
        item_data['style_features'] = extract_textual_style_features(
            item_data.get('name'), 
            item_data.get('category'), 
            item_data.get('tags')
        )
    except Exception as e:
        logger.error(f"Error extracting textual style features: {e}")


    db_item = models.WardrobeItem(
        **item_data,
        user_id=current_user.id,
        date_added=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        times_worn=0,
        favorite=item_data.get('favorite', False)
    )
    
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    # If item is created as favorite, update style profile
    if db_item.favorite:
        try:
            user_style_profile_service.update_style_profile_from_item_interaction(
                db, current_user.id, db_item.id, "favorite"
            )
            db.commit() # Commit changes to UserStyleProfile
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating style profile after creating favorite item {db_item.id} for user {current_user.id}: {e}")

    return db_item

@router.get("/items/", response_model=List[schemas.WardrobeItem])
async def read_wardrobe_items(
    category: Optional[str] = None,
    season: Optional[str] = None,
    favorite: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    query = db.query(models.WardrobeItem).filter(models.WardrobeItem.user_id == current_user.id)

    if category:
        query = query.filter(models.WardrobeItem.category.ilike(f"%{category}%"))
    if season:
        query = query.filter(models.WardrobeItem.season.ilike(f"%{season}%"))
    if favorite is not None:
        query = query.filter(models.WardrobeItem.favorite == favorite)

    items = query.offset(skip).limit(limit).all()
    return items

@router.get("/items/{item_id}", response_model=schemas.WardrobeItem)
async def read_wardrobe_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    db_item = db.query(models.WardrobeItem).filter(models.WardrobeItem.id == item_id, models.WardrobeItem.user_id == current_user.id).first()
    if db_item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    return db_item

@router.put("/items/{item_id}", response_model=schemas.WardrobeItem)
async def update_wardrobe_item(
    item_id: int,
    item_update: schemas.WardrobeItemUpdate = Depends(),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    db_item = db.query(models.WardrobeItem).filter(models.WardrobeItem.id == item_id, models.WardrobeItem.user_id == current_user.id).first()
    if db_item is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    
    update_data = item_update.model_dump(exclude_unset=True)
    previous_favorite_status = db_item.favorite
    new_favorite_status = update_data.get('favorite', previous_favorite_status)

    if image and image.filename:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image type. Allowed types: {ALLOWED_CONTENT_TYPES}")

        image_bytes_content = await image.read()
        await image.seek(0)

        if len(image_bytes_content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Image too large. Max size: {MAX_FILE_SIZE_BYTES // (1024*1024)}MB")

        if db_item.image_url:
            old_image_path_on_disk = db_item.image_url.lstrip("/")
            if os.path.exists(old_image_path_on_disk):
                try: os.remove(old_image_path_on_disk)
                except Exception as e: logger.error(f"Error deleting old image {old_image_path_on_disk}: {e}")

        unique_id = uuid.uuid4()
        extension = os.path.splitext(image.filename)[1]
        new_filename = f"{unique_id}{extension}"
        new_file_path_on_disk = os.path.join(WARDROBE_IMAGES_DIR, new_filename)

        try:
            with open(new_file_path_on_disk, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            update_data['image_url'] = f"/{new_file_path_on_disk}"

            pil_image_for_embedding = Image.open(io.BytesIO(image_bytes_content))

            try: update_data['ai_embedding'] = ai_embedding.get_image_embedding(pil_image_for_embedding)
            except Exception as e: logger.error(f"Error generating image embedding for update: {e}")

            try:
                color_detector = get_color_detector()
                dominant_color_info = color_detector.extract_dominant_color(image_bytes_content)
                palette_info = color_detector.extract_color_palette(image_bytes_content)
                if dominant_color_info.get("success"):
                    update_data['dominant_color_rgb'] = dominant_color_info["dominant_color"]["rgb"]
                    update_data['dominant_color_hex'] = dominant_color_info["dominant_color"]["hex"]
                    update_data['dominant_color_name'] = dominant_color_info["dominant_color"]["name"]
                    update_data['color_properties'] = dominant_color_info["properties"]
                if palette_info.get("success"):
                    update_data['color_palette'] = palette_info["palette"]
                    update_data['ai_dominant_colors'] = [p_color['hex'] for p_color in palette_info["palette"]] if palette_info["palette"] else []
                else:
                    update_data['ai_dominant_colors'] = [dominant_color_info["dominant_color"]["hex"]] if dominant_color_info.get("success") else []

            except Exception as e: logger.error(f"Error processing colors with ColorDetector for update: {e}")

            try:
                classifier = get_clothing_classifier()
                classification_result = classifier.classify_clothing_item(image_bytes_content)
                if classification_result.get("success"):
                    update_data['ai_classification'] = {
                        "category": classification_result["clothing_type"],
                        "confidence": classification_result["confidence"],
                        "details": classification_result["detailed_predictions"]
                    }
            except Exception as e: logger.error(f"Error classifying clothing for update: {e}")

        except Exception as e: logger.error(f"Error saving new image or during AI processing for update: {e}")
        finally: image.file.close()

    elif 'image_url' in update_data and update_data['image_url'] is None:
        if db_item.image_url:
            old_image_path_on_disk = db_item.image_url.lstrip("/")
            if os.path.exists(old_image_path_on_disk):
                try: os.remove(old_image_path_on_disk)
                except Exception as e: logger.error(f"Error deleting image {old_image_path_on_disk}: {e}")
        
        update_data.update({
            'ai_embedding': None, 'ai_dominant_colors': None, 'dominant_color_rgb': None,
            'dominant_color_hex': None, 'dominant_color_name': None, 'color_palette': None,
            'color_properties': None, 'ai_classification': None
            # style_features might be kept if purely text-based, or cleared if image-dependent
        })


    # Update style_features based on potentially updated textual data (name, category, tags)
    # This should happen whether a new image is uploaded or not, if textual fields changed.
    current_name = update_data.get('name', db_item.name)
    current_category = update_data.get('category', db_item.category)
    current_tags = update_data.get('tags', db_item.tags) # db_item.tags is already a list
    
    try:
        update_data['style_features'] = extract_textual_style_features(current_name, current_category, current_tags)
    except Exception as e:
        logger.error(f"Error extracting textual style features for update: {e}")


    for key, value in update_data.items():
        setattr(db_item, key, value)

    db_item.updated_at = datetime.utcnow()
    
    try:
        db.commit() # Commit item changes first
        db.refresh(db_item)
        # Update style profile if favorite status changed
        if new_favorite_status != previous_favorite_status:
            interaction = "favorite" if new_favorite_status else "unfavorite"
            user_style_profile_service.update_style_profile_from_item_interaction(
                db, current_user.id, db_item.id, interaction
            )
            db.commit() # Commit UserStyleProfile changes
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating item {item_id} or its style profile interaction for user {current_user.id}: {e}")
        # Potentially re-raise or handle, for now, the item update might be partial if error is in style profile part
        raise HTTPException(status_code=500, detail=f"Error processing item update or style profile: {str(e)}")

    return db_item

@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_wardrobe_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    db_item = db.query(models.WardrobeItem).filter(models.WardrobeItem.id == item_id, models.WardrobeItem.user_id == current_user.id).first()

    if db_item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")

    if db_item.image_url:
        image_path_on_disk = db_item.image_url.lstrip("/")
        if os.path.exists(image_path_on_disk):
            try: os.remove(image_path_on_disk)
            except FileNotFoundError: logger.warning(f"File not found: {image_path_on_disk}")
            except Exception as e: logger.error(f"Error deleting file {image_path_on_disk}: {e}")
        else:
            logger.warning(f"Image path not found, but listed in DB: {image_path_on_disk}")

    db.delete(db_item)
    db.commit()
    return

# Endpoint for Find Matching Clothing Items (Objective ii)
@router.post("/items/{item_id}/find-matches", response_model=schemas_ml.FindMatchesForItemResponse)
async def find_matching_clothing_items(
    request_body: schemas_ml.FindMatchesForItemBodyRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
      item_id: int = Path(..., description="The ID of the wardrobe item to find matches for.")
):
    """
    Finds and suggests complementary clothing items for a given target item from the user's wardrobe.

    Suggestions are based on color coordination, style similarity, and category compatibility.
    Users can optionally specify an occasion or preferred categories for suggestions.
    The suggestions are influenced by learned user preferences over time.
    """
  

    recommendation_engine = get_recommendation_engine()

    # 1. Fetch the target WardrobeItem
    target_db_item = db.query(models.WardrobeItem).filter(
        models.WardrobeItem.id == item_id,
        models.WardrobeItem.user_id == current_user.id
    ).first()

    if not target_db_item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target wardrobe item not found.")

    # Convert target item to Pydantic schema for the response
    target_item_response = schemas_ml.WardrobeItemResponse.model_validate(target_db_item)


    # 2. Fetch all other items for the user to consider as potential matches
    potential_match_items_db = db.query(models.WardrobeItem).filter(
        models.WardrobeItem.user_id == current_user.id,
        models.WardrobeItem.id != item_id
    ).all()

    if not potential_match_items_db:
        return schemas_ml.FindMatchesForItemResponse(
            target_item=target_item_response,
            suggested_matches=[],
            message="No other items in the wardrobe to suggest matches from."
        )

    # Convert DB items to dictionaries suitable for recommendation engine
    # The engine expects dicts with specific keys like 'dominant_color', 'style', 'tags'
    def format_item_for_engine(db_item: models.WardrobeItem) -> Dict[str, Any]:
        # This should align with how features are prepared in OutfitRecommendationEngine
        # or how its methods expect item data.
        return {
            "id": db_item.id, # Important for tracking
            "name": db_item.name,
            "category": db_item.category,
            "brand": db_item.brand,
            # Dominant color: engine might expect name or RGB. Let's provide name if available.
            "dominant_color": {"name": db_item.dominant_color_name, "rgb": db_item.dominant_color_rgb} if db_item.dominant_color_name else None,
            "style_features": db_item.style_features if db_item.style_features else {"identified_styles": [], "raw_keywords": []},
            "tags": db_item.tags, # Already a list from the property
            "image_url": db_item.image_url,
            "description": f"{db_item.name} {db_item.category} {' '.join(db_item.tags if db_item.tags else [])}", # Simple text for style extraction
            # Add other fields the engine might use: price, material, season etc.
            "price": db_item.price,
            "material": db_item.material,
            "season": db_item.season,
            "ai_embedding": db_item.ai_embedding # Crucial for embedding-based similarity
        }

    formatted_target_item = format_item_for_engine(target_db_item)
    formatted_potential_matches = [format_item_for_engine(pi) for pi in potential_match_items_db]
    
    suggested_matches_map: Dict[int, schemas_ml.SuggestedItemMatch] = {}

    # 3.a. Get color-coordinated items
    color_matches = recommendation_engine.get_color_coordinated_items(formatted_target_item, formatted_potential_matches)
    for match_dict in color_matches:
        item_id = match_dict["id"]
        if item_id not in suggested_matches_map:
            # Find the original DB item to construct WardrobeItemResponse
            original_item = next((i for i in potential_match_items_db if i.id == item_id), None)
            if original_item:
                suggested_matches_map[item_id] = schemas_ml.SuggestedItemMatch(
                    item=schemas_ml.WardrobeItemResponse.model_validate(original_item),
                    match_score=match_dict.get("color_compatibility_score", 0.7), # Default score
                    match_type="color_coordinated",
                    reason=f"Coordinates well by color with {target_item_response.name}."
                )
    
    # 3.b. Get similar items (style, category etc.)
    # Note: OutfitRecommendationEngine.get_similar_items uses its own feature prep.
    # We might need to ensure the engine is trained or can work with raw dicts.
    # For now, assume it can take these formatted dicts.
    # If the engine needs to be trained first, this endpoint should check that.
    # Let's assume it's ready or uses a pre-trained model / direct feature comparison.
    
    # The engine's get_similar_items expects the target item and list of other items.
    # Its internal feature prep should handle these dicts.
    similar_style_matches = recommendation_engine.get_similar_items(formatted_target_item, formatted_potential_matches, num_similar=request_body.num_suggestions * 2) # Get more to filter later

    for match_dict in similar_style_matches:
        item_id = match_dict["id"]
        similarity_score = match_dict.get("similarity_score", 0.0)
        
        original_item = next((i for i in potential_match_items_db if i.id == item_id), None)
        if not original_item:
            continue

        if item_id in suggested_matches_map:
            # Item already suggested by color match, update score if this is higher, add reason
            if similarity_score > suggested_matches_map[item_id].match_score:
                suggested_matches_map[item_id].match_score = similarity_score
            suggested_matches_map[item_id].match_type += ", style_similar"
            suggested_matches_map[item_id].reason += f" Also similar in style/features to {target_item_response.name}."
        else:
             suggested_matches_map[item_id] = schemas_ml.SuggestedItemMatch(
                item=schemas_ml.WardrobeItemResponse.model_validate(original_item),
                match_score=similarity_score,
                match_type="style_similar",
                reason=f"Similar style/features to {target_item_response.name}."
            )

    # 4. Filter suggestions
    
    # Define complementary categories
    COMPLEMENTARY_CATEGORIES = {
        "tops": ["bottoms", "pants", "skirts", "shorts", "outerwear"],
        "blouses": ["bottoms", "pants", "skirts", "shorts", "outerwear"],
        "shirts": ["bottoms", "pants", "skirts", "shorts", "outerwear"],
        "t-shirts": ["bottoms", "pants", "skirts", "shorts", "outerwear"],
        "sweaters": ["bottoms", "pants", "skirts", "shorts", "outerwear"],
        "bottoms": ["tops", "blouses", "shirts", "t-shirts", "sweaters", "outerwear"],
        "pants": ["tops", "blouses", "shirts", "t-shirts", "sweaters", "outerwear"],
        "jeans": ["tops", "blouses", "shirts", "t-shirts", "sweaters", "outerwear"],
        "skirts": ["tops", "blouses", "shirts", "t-shirts", "sweaters", "outerwear"],
        "shorts": ["tops", "blouses", "shirts", "t-shirts", "sweaters"],
        "dresses": ["outerwear", "shoes", "accessories"],
        "outerwear": ["tops", "blouses", "shirts", "t-shirts", "sweaters", "dresses", "bottoms", "pants", "skirts"],
        "shoes": [], # Shoes typically don't "match" other shoes as primary items
        "accessories": [], # Accessories are complements, not primary matches usually
    }

    final_suggestions: List[schemas_ml.SuggestedItemMatch] = []
    
    target_category_lower = target_item_response.category.lower()
    
    # Determine categories to suggest
    categories_to_actively_suggest = set()
    if request_body.suggest_categories:
        categories_to_actively_suggest.update([cat.lower() for cat in request_body.suggest_categories])
    else: # Default to complementary categories
        categories_to_actively_suggest.update(COMPLEMENTARY_CATEGORIES.get(target_category_lower, []))
        # If no specific complementary categories (e.g. for shoes), suggest broadly or let scores decide.
        # For now, if target_category_lower is not in COMPLEMENTARY_CATEGORIES or its list is empty,
        # it means we don't have a strong default suggestion list, so all categories are fair game initially.
        if not categories_to_actively_suggest: # E.g. target is shoes, or unknown category
             # Allow any category if no specific suggestions or target is accessory-like
            pass # No specific category filtering, rely on scores.


    # Filter and sort
    # Convert map to list first
    all_potential_suggestions = sorted(list(suggested_matches_map.values()), key=lambda x: x.match_score, reverse=True)
    
    for suggested_match in all_potential_suggestions:
        if len(final_suggestions) >= request_body.num_suggestions:
            break

        # Category filtering
        if categories_to_actively_suggest: # If we have a list of preferred categories
            if suggested_match.item.category.lower() not in categories_to_actively_suggest:
                continue # Skip if not in the desired list of categories
        elif target_category_lower == suggested_match.item.category.lower() and target_category_lower not in ["accessories", "shoes"]:
            # Avoid suggesting same category unless it's an accessory or shoe (where multiple might make sense)
            # Or unless user explicitly asked for this category via request_body.suggest_categories (covered by above if)
            continue
            
        # TODO: Occasion filtering if request_body.occasion is provided
        # This would involve checking item.season, item.style_features against occasion.
        # Example: if request_body.occasion == "formal", check if item.style_features["identified_styles"] contains "formal".
        # This logic can be complex and might reside in the recommendation engine. For now, basic.
        if request_body.occasion:
            item_styles = suggested_match.item.style_features.get("identified_styles", []) if suggested_match.item.style_features else []
            # Simple check: if occasion is "formal", item style should include "formal"
            # This is very basic. A more robust check would use occasion_appropriateness from engine.
            if request_body.occasion.lower() in ["formal", "business", "party"] and not any(s.lower() == request_body.occasion.lower() for s in item_styles):
                # If a specific occasion is requested, and the item's styles don't seem to match,
                # we might lower its score or skip it. For now, a simple skip if no direct style match.
                # This needs refinement.
                # continue # Tentatively disabled to not filter too aggressively initially
                pass


        # TODO: Integrate UserStyleProfile for personalization
        # - Boost score if item matches UserStyleProfile.preferred_colors, categories, styles, brands.
        # - Penalize if item matches avoided colors.
        # This requires fetching UserStyleProfile for current_user.

        final_suggestions.append(suggested_match)


    return schemas_ml.FindMatchesForItemResponse(
        target_item=target_item_response,
        suggested_matches=final_suggestions,
        message=f"Found {len(final_suggestions)} suggestions for {target_item_response.name}."
    )
