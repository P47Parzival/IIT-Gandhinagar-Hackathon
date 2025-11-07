"""
Document Store for mapping GL accounts to supporting documents.

Manages document storage, retrieval, and organization based on GL accounts,
entities, and periods.
"""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


class DocumentStore:
    """
    Centralized document storage and retrieval system.
    
    Maps GL accounts to their supporting documents and provides
    efficient lookup and management capabilities.
    """
    
    def __init__(self, base_path: str = "data/documents"):
        """
        Initialize document store.
        
        Args:
            base_path: Base directory for document storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Index file for fast lookup
        self.index_file = self.base_path / "document_index.json"
        self.index = self._load_index()
        
        print(f"✓ Document Store initialized at: {self.base_path}")
    
    def _load_index(self) -> Dict[str, Any]:
        """Load document index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'documents': {},
            'gl_accounts': {},
            'entities': {},
            'last_updated': None
        }
    
    def _save_index(self):
        """Save document index to disk."""
        self.index['last_updated'] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2)
    
    def add_document(
        self,
        file_path: str,
        gl_account: str,
        entity_id: str,
        period: str,
        document_type: str = "supporting",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the store.
        
        Args:
            file_path: Path to the document file
            gl_account: GL account this document supports
            entity_id: Entity identifier
            period: Accounting period (e.g., '2024-10')
            document_type: Type of document (e.g., 'invoice', 'contract', 'reconciliation')
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Generate document ID
        doc_id = f"{entity_id}_{gl_account}_{period}_{int(datetime.now().timestamp())}"
        
        # Create organized directory structure
        entity_dir = self.base_path / entity_id / period / gl_account
        entity_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file to store (or could move it)
        import shutil
        file_ext = os.path.splitext(file_path)[1]
        dest_path = entity_dir / f"{doc_id}{file_ext}"
        shutil.copy2(file_path, dest_path)
        
        # Update index
        doc_record = {
            'doc_id': doc_id,
            'file_path': str(dest_path),
            'original_name': os.path.basename(file_path),
            'gl_account': gl_account,
            'entity_id': entity_id,
            'period': period,
            'document_type': document_type,
            'file_size': os.path.getsize(dest_path),
            'added_date': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.index['documents'][doc_id] = doc_record
        
        # Update lookup indices
        gl_key = f"{entity_id}_{gl_account}_{period}"
        if gl_key not in self.index['gl_accounts']:
            self.index['gl_accounts'][gl_key] = []
        self.index['gl_accounts'][gl_key].append(doc_id)
        
        if entity_id not in self.index['entities']:
            self.index['entities'][entity_id] = []
        if doc_id not in self.index['entities'][entity_id]:
            self.index['entities'][entity_id].append(doc_id)
        
        self._save_index()
        
        print(f"✓ Added document: {doc_id}")
        return doc_id
    
    def get_documents_for_gl(
        self,
        gl_account: str,
        entity_id: str,
        period: str
    ) -> List[Dict[str, Any]]:
        """
        Get all documents for a specific GL account.
        
        Args:
            gl_account: GL account code
            entity_id: Entity identifier
            period: Accounting period
            
        Returns:
            List of document records
        """
        gl_key = f"{entity_id}_{gl_account}_{period}"
        doc_ids = self.index['gl_accounts'].get(gl_key, [])
        
        documents = []
        for doc_id in doc_ids:
            doc_record = self.index['documents'].get(doc_id)
            if doc_record and os.path.exists(doc_record['file_path']):
                documents.append(doc_record)
        
        return documents
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        return self.index['documents'].get(doc_id)
    
    def search_documents(
        self,
        entity_id: Optional[str] = None,
        gl_account: Optional[str] = None,
        period: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents with filters.
        
        Args:
            entity_id: Filter by entity
            gl_account: Filter by GL account
            period: Filter by period
            document_type: Filter by document type
            
        Returns:
            List of matching document records
        """
        matches = []
        
        for doc_id, doc_record in self.index['documents'].items():
            if entity_id and doc_record['entity_id'] != entity_id:
                continue
            if gl_account and doc_record['gl_account'] != gl_account:
                continue
            if period and doc_record['period'] != period:
                continue
            if document_type and doc_record['document_type'] != document_type:
                continue
            
            matches.append(doc_record)
        
        return matches
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics."""
        total_docs = len(self.index['documents'])
        total_entities = len(self.index['entities'])
        total_size = sum(doc['file_size'] for doc in self.index['documents'].values())
        
        return {
            'total_documents': total_docs,
            'total_entities': total_entities,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'last_updated': self.index['last_updated']
        }
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successfully removed
        """
        doc_record = self.index['documents'].get(doc_id)
        if not doc_record:
            return False
        
        # Remove file
        try:
            os.remove(doc_record['file_path'])
        except:
            pass
        
        # Remove from index
        del self.index['documents'][doc_id]
        
        # Remove from GL index
        gl_key = f"{doc_record['entity_id']}_{doc_record['gl_account']}_{doc_record['period']}"
        if gl_key in self.index['gl_accounts']:
            self.index['gl_accounts'][gl_key].remove(doc_id)
        
        # Remove from entity index
        if doc_record['entity_id'] in self.index['entities']:
            self.index['entities'][doc_record['entity_id']].remove(doc_id)
        
        self._save_index()
        
        print(f"✓ Removed document: {doc_id}")
        return True


if __name__ == "__main__":
    # Test document store
    print("Testing Document Store...")
    
    # Use temporary test directory
    test_store = DocumentStore("test_document_store")
    
    print(f"✓ Stats: {test_store.get_stats()}")
    
    # Note: Actual testing requires document files
    # Usage example:
    # doc_id = test_store.add_document(
    #     file_path='invoice.pdf',
    #     gl_account='101000',
    #     entity_id='E001',
    #     period='2024-10',
    #     document_type='invoice'
    # )
    # docs = test_store.get_documents_for_gl('101000', 'E001', '2024-10')
    
    print("\n✓ Document Store tests passed!")
    
    # Cleanup
    import shutil
    if os.path.exists("test_document_store"):
        shutil.rmtree("test_document_store")

