class ArchetypeManager:
    @staticmethod
    def get_config(archetype):
        import utils.utils as utils
        return utils.DictToAttribute(**utils.load_yaml('archetypes.yaml')['agent_archetypes'][archetype])

    @staticmethod
    def get_name(archetype):
        return ArchetypeManager.get_config(archetype).name
