from bson.objectid import ObjectId
from datetime import datetime
from copy import deepcopy
from ml_forest.core.utils.docs_init import root_database


class DocsHandler(object):
    def __init__(self):
        pass

    def init_doc(self, obj, update_dict=True):
        """
        The "essentials" attribute of an obj would be used to identify the obj from the db.

        :param obj:
        :param update_dict: bool. If the training is documented in a dictionary locally, this allow users to decide
                                  if the pickled documents are to be updated in this function call.
        :return:
        """
        try:
            obj.essentials
        except AttributeError:
            raise AttributeError("An object to be saved in db is supposed to have the essentials attribute")

        if obj.essentials is None:
            raise AttributeError("An object to be saved in db should not have NoneType as its essentials")

        print("Saving this object into db: {}".format(type(obj)))

        start = datetime.now()
        essen = obj.essentials
        document = {"essentials": essen, 'datetime': start, 'filepaths': obj.filepaths}

        db_location = obj.db
        element = obj.decide_element()
        project = db_location["project"]

        target_collection = root_database[project][element]
        inserted_id = ObjectId()
        document["_id"] = inserted_id
        target_collection.append(document)

        return inserted_id

    @staticmethod
    def insert_tag(obj, tag):
        qry = {"tag": tag}
        DocsHandler.update_doc(obj, qry)

    @staticmethod
    def update_doc(obj, qry):
        if not isinstance(qry, dict):
            raise TypeError("The new updating query should be encoded into a dictionary.")
        if not obj.obj_id:
            raise AttributeError("The obj passed has no obj_id attribute, can't find the document.")
        obj_id = obj.obj_id

        try:
            db_location = obj.db
        except AttributeError:
            raise AttributeError("The obj passed has no db attribute, can't find the location of the document.")

        try:
            element = obj.decide_element()
        except AttributeError:
            msg = "The object passed has no decide_element method. Is this object originally designed to be tracked?"
            raise AttributeError(msg)

        pt2doc = DocsHandler.pt2doc_by_obj_id(obj_id, element, db_location)

        for key in qry:
            pt2doc[key] = qry[key]

    @staticmethod
    def pt2doc_by_obj_id(obj_id, element, db):
        project = db["project"]
        target_collection = root_database[project][element]

        found = [d for d in target_collection if d["_id"]==obj_id]
        if len(found)> 1:
            raise ValueError("There are more than one document with the objectid you passed")

        pt2doc = found[0]
        return pt2doc

    @staticmethod
    def search_by_obj_id(obj_id, element, db):
        doc = DocsHandler.pt2doc_by_obj_id(obj_id, element, db)
        result = deepcopy(doc)

        return result

    @staticmethod
    def insert_subdoc_by_id(obj_id, element, db, field, subdoc):
        pt2doc = DocsHandler.pt2doc_by_obj_id(obj_id, element, db)

        if not isinstance(field, str):
            raise ValueError("A field has to be a string")

        subdoc = deepcopy(subdoc)
        pt2doc[field].append(subdoc)

    @staticmethod
    def pt2doc_by_essentials(obj, db):
        project = db["project"]
        element = obj.decide_element()
        target_collection = root_database[project][element]

        essen = deepcopy(obj.essentials)
        found = [d for d in target_collection if d["essentials"] == essen]

        if not found:
            return []
        else:
            pt2doc = found
            return pt2doc

    @staticmethod
    def search_by_essentials(obj, db):
        doc = DocsHandler.pt2doc_by_essentials(obj, db)
        result = deepcopy(doc)

        return result

    @staticmethod
    def search_obj_by_tag(tag, element, db):
        project = db["project"]
        target_collection = root_database[project][element]

        found = [d for d in target_collection if "tag" in d and d["tag"]==tag]

        pt2doc = found[0]
        return pt2doc

    @staticmethod
    def search_core_by_tag(tag, db):
        result = DocsHandler.search_obj_by_tag(tag, "CoreInit", db)

        return result

    @staticmethod
    def delete_by_lst_obj_id(lst_obj_id, element, db):
        project = db["project"]
        target_collection = root_database[project][element]

        for obj_id in lst_obj_id:
            found = [d for d in target_collection if d["_id"]==obj_id][0]
            target_collection.remove(found)
