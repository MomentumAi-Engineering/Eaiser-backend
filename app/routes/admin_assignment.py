"""
Issue Assignment Endpoint
"""

@router.post("/assign-issue")
async def assign_issue_to_admin(
    issue_id: str,
    admin_email: str,
    current_admin: dict = Depends(get_admin_user)
):
    """
    Assign an issue to a specific admin. Only super_admin can assign.
    """
    # Check permissions
    if current_admin.get("role") != "super_admin":
        raise HTTPException(
            status_code=403,
            detail="Only super admins can assign issues"
        )
    
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        # Find the admin to assign to
        admins_collection = await mongo_service.get_collection("admins")
        target_admin = await admins_collection.find_one({"email": admin_email})
        
        if not target_admin:
            raise HTTPException(status_code=404, detail=f"Admin {admin_email} not found")
        
        if not target_admin.get("is_active"):
            raise HTTPException(status_code=400, detail="Cannot assign to inactive admin")
        
        # Check if admin has permission to handle issues
        permissions = target_admin.get("permissions", {})
        if not permissions.get("can_approve") and not permissions.get("can_decline"):
            raise HTTPException(
                status_code=400,
                detail=f"Admin {admin_email} (role: {target_admin.get('role')}) cannot handle issues"
            )
        
        # Update admin's assigned_issues list
        await admins_collection.update_one(
            {"email": admin_email},
            {"$addToSet": {"assigned_issues": issue_id}}
        )
        
        # Update issue with assigned_to field
        issues_collection = await mongo_service.get_collection("issues")
        await issues_collection.update_one(
            {"_id": issue_id},
            {
                "$set": {
                    "assigned_to": admin_email,
                    "assigned_by": current_admin.get("email"),
                    "assigned_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"✅ Issue {issue_id} assigned to {admin_email} by {current_admin.get('email')}")
        
        return {
            "message": f"Issue assigned to {target_admin.get('name', admin_email)}",
            "issue_id": issue_id,
            "assigned_to": {
                "email": admin_email,
                "name": target_admin.get("name"),
                "role": target_admin.get("role")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning issue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/my-assigned-issues")
async def get_my_assigned_issues(current_admin: dict = Depends(get_admin_user)):
    """
    Get all issues assigned to the current admin
    """
    try:
        mongo_service = await get_optimized_mongodb_service()
        if not mongo_service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        admin_email = current_admin.get("email")
        
        # Get admin's assigned issues
        issues_collection = await mongo_service.get_collection("issues")
        cursor = issues_collection.find({"assigned_to": admin_email}).sort("timestamp", -1)
        assigned_issues = await cursor.to_list(length=100)
        
        # Format response
        for issue in assigned_issues:
            if "_id" in issue:
                issue["_id"] = str(issue["_id"])
                issue["issue_id"] = issue["_id"]
        
        logger.info(f"📋 Retrieved {len(assigned_issues)} assigned issues for {admin_email}")
        
        return {
            "count": len(assigned_issues),
            "issues": assigned_issues
        }
        
    except Exception as e:
        logger.error(f"Error fetching assigned issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))
